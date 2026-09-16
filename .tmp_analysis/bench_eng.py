"""Measure the engineering issues: mask construction cost, activation memory, grad checkpointing.

Run on GPU:
  PYTHONPATH=. python .tmp_analysis/bench_eng.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ContextRecord, QARecord, collate_fn  # noqa: E402
from src.losses import qa_loss, reconstruction_loss  # noqa: E402
from src.model import build_block_causal_mask, build_continuation_mask, load_model  # noqa: E402
from utils.config import TrainConfig  # noqa: E402

DEV = torch.device("cuda:0")


# ---------------------------------------------------------------- vectorised masks
def continuation_mask_fast(question_mask, answer_mask, memory_length, dtype):
    q, a = question_mask.bool(), answer_mask.bool()
    b, ql = q.shape
    al = a.shape[1]
    cur = ql + al
    total = memory_length + cur
    idx = torch.arange(cur, device=q.device)
    causal = idx[:, None] >= idx[None, :]
    allowed = torch.zeros(b, cur, total, dtype=torch.bool, device=q.device)
    allowed[:, :, :memory_length] = True
    allowed[:, :ql, memory_length:memory_length + ql] = q[:, None, :] & causal[:ql, :ql]
    allowed[:, ql:, memory_length:memory_length + ql] = q[:, None, :]
    allowed[:, ql:, memory_length + ql:] = a[:, None, :] & causal[ql:, ql:]
    valid = torch.cat([q, a], dim=1)
    self_edge = torch.zeros_like(allowed)
    self_edge[:, idx, memory_length + idx] = True
    allowed = torch.where(valid[:, :, None], allowed, self_edge)
    mask = torch.zeros(b, 1, cur, total, dtype=dtype, device=q.device)
    return mask.masked_fill(~allowed[:, None], torch.finfo(dtype).min)


def block_mask_fast(context_mask, memory_length, question_mask, answer_mask, dtype):
    c, q, a = context_mask.bool(), question_mask.bool(), answer_mask.bool()
    b, length = c.shape
    ql, al = q.shape[1], a.shape[1]
    total = length + memory_length + ql + al
    mem_start = length
    q_start = length + memory_length
    a_start = q_start + ql
    dev = c.device
    allowed = torch.zeros(b, total, total, dtype=torch.bool, device=dev)
    # context rows: causal over valid context tokens only
    allowed[:, :length, :length] = c[:, None, :] & torch.tril(
        torch.ones(length, length, dtype=torch.bool, device=dev))
    # memory rows: every valid context token, nothing else (no memory<->memory)
    allowed[:, mem_start:q_start, :length] = c[:, None, :]
    qa_causal = torch.tril(torch.ones(ql + al, ql + al, dtype=torch.bool, device=dev))
    # question rows: all memory + causal over valid question tokens
    allowed[:, q_start:a_start, mem_start:q_start] = True
    allowed[:, q_start:a_start, q_start:a_start] = q[:, None, :] & qa_causal[:ql, :ql]
    # answer rows: all memory + all valid question tokens + causal over valid answer tokens
    allowed[:, a_start:, mem_start:q_start] = True
    allowed[:, a_start:, q_start:a_start] = q[:, None, :]
    allowed[:, a_start:, a_start:] = a[:, None, :] & qa_causal[ql:, ql:]
    # padding query rows keep a single self edge
    rows = torch.arange(length, device=dev)
    ctx_eye = torch.zeros(b, length, total, dtype=torch.bool, device=dev)
    ctx_eye[:, rows, rows] = True
    allowed[:, :length] = torch.where(c[:, :, None], allowed[:, :length], ctx_eye)
    qa_rows = torch.arange(ql + al, device=dev)
    qa_eye = torch.zeros(b, ql + al, total, dtype=torch.bool, device=dev)
    qa_eye[:, qa_rows, q_start + qa_rows] = True
    qa_valid = torch.cat([q, a], dim=1)
    allowed[:, q_start:] = torch.where(qa_valid[:, :, None], allowed[:, q_start:], qa_eye)
    mask = torch.zeros(b, 1, total, total, dtype=dtype, device=dev)
    return mask.masked_fill(~allowed[:, None], torch.finfo(dtype).min)


def correctness():
    torch.manual_seed(0)
    ok = True
    for trial in range(20):
        b = 3
        length, memory, ql, al = 17, 4, 5, 6
        c = torch.rand(b, length, device=DEV) > 0.3
        q = torch.rand(b, ql, device=DEV) > 0.4
        a = torch.rand(b, al, device=DEV) > 0.4
        for row in range(b):  # guarantee at least one valid question/answer row
            q[row, 0] = True
            a[row, 0] = True
        ref_b = build_block_causal_mask(c, memory, q, a, torch.bfloat16)
        new_b = block_mask_fast(c, memory, q, a, torch.bfloat16)
        ref_c = build_continuation_mask(q, a, memory, torch.bfloat16)
        new_c = continuation_mask_fast(q, a, memory, torch.bfloat16)
        ok &= bool(torch.equal(ref_b, new_b)) and bool(torch.equal(ref_c, new_c))
        if not (torch.equal(ref_b, new_b) and torch.equal(ref_c, new_c)):
            print("  MISMATCH at trial", trial)
            break
    print("mask correctness (20 random batches, exact equality):", "PASS" if ok else "FAIL", flush=True)


def timing():
    torch.cuda.synchronize()
    cases = [
        ("block  B=2 L=2048 M=8 QL=64 AL=32", lambda: build_block_causal_mask(
            torch.ones(2, 2048, dtype=torch.bool, device=DEV), 8,
            torch.ones(2, 64, dtype=torch.bool, device=DEV),
            torch.ones(2, 32, dtype=torch.bool, device=DEV), torch.bfloat16)),
        ("block  B=2 L=2048 M=8 QL=64 AL=32  FAST", lambda: block_mask_fast(
            torch.ones(2, 2048, dtype=torch.bool, device=DEV), 8,
            torch.ones(2, 64, dtype=torch.bool, device=DEV),
            torch.ones(2, 32, dtype=torch.bool, device=DEV), torch.bfloat16)),
        ("cont   B=8 M=8 QL=64 AL=32", lambda: build_continuation_mask(
            torch.ones(8, 64, dtype=torch.bool, device=DEV),
            torch.ones(8, 32, dtype=torch.bool, device=DEV), 8, torch.bfloat16)),
        ("cont   B=8 M=8 QL=64 AL=32  FAST", lambda: continuation_mask_fast(
            torch.ones(8, 64, dtype=torch.bool, device=DEV),
            torch.ones(8, 32, dtype=torch.bool, device=DEV), 8, torch.bfloat16)),
    ]
    for name, fn in cases:
        fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        print("  %-40s %.2f ms/call" % (name, (time.time() - t0) / 20 * 1000), flush=True)


# ---------------------------------------------------------------- training-step memory
def make_batch(cfg, tokenizer, n_ctx, ctx_tokens):
    text = " ".join(["word%d" % i for i in range(ctx_tokens)])
    rows = [ContextRecord(text, (QARecord("q%d?" % i, "answer%d" % i, "x", ""),
                                 QARecord("qq%d?" % i, "answer%d" % i, "x", ""),
                                 QARecord("qqq%d?" % i, "answer%d" % i, "x", ""),
                                 QARecord("qqqq%d?" % i, "answer%d" % i, "x", "")) , "x", "")
            for i in range(n_ctx)]
    return collate_fn(rows, tokenizer, cfg.data.max_context_tokens, cfg.data.max_question_tokens,
                      cfg.data.max_answer_tokens, cfg.data.append_eos, False, False,
                      cfg.data.qa_per_context, True, "left", "append")


def step_memory(cfg, tokenizer, n_ctx, ctx_tokens, use_recon=True, ckpt=False):
    model = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        _, model = load_model(cfg)
        model.to(DEV).train()
        if ckpt:
            model.qwen.gradient_checkpointing_enable()
            model.qwen.enable_input_require_grads()
        batch = make_batch(cfg, tokenizer, n_ctx, ctx_tokens)
        ids = {k: v.to(DEV) for k, v in batch.items() if k != "records"}
        emb = model.qwen.get_input_embeddings()
        out = model(emb(ids["context_ids"]), ids["context_mask"], emb(ids["question_ids"]),
                    ids["question_mask"], emb(ids["answer_ids"]), ids["answer_mask"],
                    ids["labels"], ids["qa_context_indices"])
        loss = qa_loss(out.logits, out.labels)
        if use_recon:
            loss = loss + reconstruction_loss(out.reconstruction, out.context_target, out.context_mask, 0.5)
        loss.backward()
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        mem_grad = model.memory_tokens.grad
        lora_grad = max((p.grad.abs().max().item() for n, p in model.named_parameters()
                         if "lora_B" in n and p.grad is not None), default=float("nan"))
        print("  ctx=%d recon=%-5s ckpt=%-5s peak=%.2f GiB  memory_grad_norm=%.4g  max|lora_B.grad|=%.3g"
              % (ctx_tokens, use_recon, ckpt, peak,
                 float(mem_grad.norm()) if mem_grad is not None else -1.0, lora_grad), flush=True)
    except torch.cuda.OutOfMemoryError:
        print("  ctx=%d recon=%-5s ckpt=%-5s OOM" % (ctx_tokens, use_recon, ckpt), flush=True)
    finally:
        del model
        torch.cuda.empty_cache()


def main():
    print("=== mask correctness", flush=True)
    correctness()
    print("=== mask timing", flush=True)
    timing()

    cfg = TrainConfig.from_file("configs/qwen-1.7b/train.yaml")
    cfg.validate()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path, local_files_only=True)
    print("=== training-step peak memory", flush=True)
    for ctx_tokens in (2048,):
        for use_recon in (True, False):
            for ckpt in (False, True):
                step_memory(cfg, tokenizer, 2, ctx_tokens, use_recon, ckpt)


if __name__ == "__main__":
    main()
