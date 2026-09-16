"""Verify two suspected evaluation bugs of the current method.

BUG 1 (question padding in the middle of the sequence)
  collate_context_records right-pads questions to the batch max, then
  model.forward_qa_with_prefix concatenates [question, answer], so for every row whose
  question is shorter than the batch max the first answer token is predicted from the
  hidden state of a *padding* position.  Autoregressive generation truncates the question
  to its valid length, so the two paths disagree on the single most important token.

BUG 2 (max_new_tokens)
  The run that reported autoregressive F1 = 0.0044 used evaluation.max_new_tokens = 128.

Usage: PYTHONPATH=. python .tmp_analysis/verify_bugs.py [N]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ContextRecord, QARecord, collate_fn  # noqa: E402
from src.icl_baseline import example_metrics  # noqa: E402
from src.model import load_model  # noqa: E402
from utils.checkpoint import CheckpointManager  # noqa: E402
from utils.config import TrainConfig  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
CKPT = sys.argv[2] if len(sys.argv) > 2 else "outputs/Qwen1.7B_20260916_024918/step-20000.pt"

cfg = TrainConfig.from_file("configs/qwen-1.7b/train.yaml")
cfg.validate()
device = torch.device("cuda:0")
tokenizer, model = load_model(cfg)
step, _ = CheckpointManager(Path(CKPT).parent).load(CKPT, model)
model.to(device).eval()
embed = model.qwen.get_input_embeddings()
print("ckpt step", step, flush=True)


def build(n, skip=0):
    out = []
    with open("/data/lz/contexts/aggregated/squad/validation.jsonl", encoding="utf-8") as fh:
        for row in fh:
            row = json.loads(row)
            for qa in row.get("qa_pairs", []) or []:
                ans = [str(a).strip() for a in (qa.get("answers") or []) if str(a).strip()]
                q = str(qa.get("question", "")).strip()
                if q and ans:
                    out.append((ContextRecord(str(row["context"]),
                                              (QARecord(q, ans[0], "squad", ""),), "squad", ""), ans))
                    break
            if len(out) >= skip + n:
                break
    return out[skip:]


def collate(chunk):
    return collate_fn(chunk, tokenizer, cfg.data.max_context_tokens, cfg.data.max_question_tokens,
                      cfg.data.max_answer_tokens, cfg.data.append_eos, False, False,
                      cfg.data.qa_per_context, False)


def teacher_forced(chunks, refs):
    """Return (project F1, official F1, first-token accuracy, n rows whose question was padded)."""
    proj, offi, hits, padded = [], [], 0, 0
    for chunk, ref in chunks:
        batch = collate(chunk)
        ids = {k: v.to(device) for k, v in batch.items() if k != "records"}
        ce, qe, ae = embed(ids["context_ids"]), embed(ids["question_ids"]), embed(ids["answer_ids"])
        with torch.no_grad():
            prefix = model.encode_context_prefix(ce, ids["context_mask"])
            out = model.forward_qa_with_prefix(
                prefix, ids["qa_context_indices"], qe, ids["question_mask"], ae, ids["answer_mask"], ids["labels"])
        pred = out.logits.argmax(-1)[:, :-1]
        labels = out.labels[:, 1:]
        qmax = int(ids["question_mask"].sum(-1).max())
        for i, (rec, refs_i) in enumerate(zip(batch["records"], ref)):
            active = labels[i].ne(-100)
            text = tokenizer.decode(pred[i][active].tolist(), skip_special_tokens=True)
            proj.append(example_metrics(text, refs_i[:1])["f1"])
            offi.append(example_metrics(text, refs_i)["f1"])
            hits += int(pred[i][active][0].item() == labels[i][active][0].item())
            padded += int(int(ids["question_mask"][i].sum()) < qmax)
        del out, prefix, ce, qe, ae, ids, batch
        torch.cuda.empty_cache()
    n = len(proj)
    return sum(proj) / n, sum(offi) / n, hits / n, padded


data = build(N)
records = [d[0] for d in data]
refs = [d[1] for d in data]

print("=" * 90)
print("BUG 1: question padding")
for chunk_size, label in ((8, "batch of 8 (questions right-padded)"), (1, "batch of 1 (no question padding)")):
    t0 = time.time()
    chunks = [(records[i:i + chunk_size], refs[i:i + chunk_size]) for i in range(0, len(records), chunk_size)]
    p, o, h, pad = teacher_forced(chunks, refs)
    print("%-42s project F1=%.4f official F1=%.4f first-token acc=%.3f  (rows with padded question=%d/%d)  %.0fs"
          % (label, p, o, h, pad, len(records), time.time() - t0), flush=True)

# ---------- controlled pair test: same row, batched alone vs batched next to a long question
print("-" * 90)
print("controlled test: identical row, only the batch composition changes")
lens = [len(tokenizer(r.qa_pairs[0].question, add_special_tokens=False).input_ids) for r in records]
short_i = min(range(len(records)), key=lambda i: lens[i])
long_i = max(range(len(records)), key=lambda i: lens[i])


def first_token_of(indices):
    chunk = [records[i] for i in indices]
    batch = collate(chunk)
    ids = {k: v.to(device) for k, v in batch.items() if k != "records"}
    with torch.no_grad():
        prefix = model.encode_context_prefix(embed(ids["context_ids"]), ids["context_mask"])
        out = model.forward_qa_with_prefix(
            prefix, ids["qa_context_indices"], embed(ids["question_ids"]), ids["question_mask"],
            embed(ids["answer_ids"]), ids["answer_mask"], ids["labels"])
    pred = out.logits.argmax(-1)[:, :-1]
    labels = out.labels[:, 1:]
    res = []
    for i in range(len(chunk)):
        active = labels[i].ne(-100)
        res.append((tokenizer.decode([pred[i][active][0].item()]),
                    tokenizer.decode([labels[i][active][0].item()]),
                    pred[i][active][0].item() == labels[i][active][0].item()))
    return res


solo = first_token_of([short_i])
paired = first_token_of([short_i, long_i])
print("short-question row: q_tokens=%d   long-question row: q_tokens=%d" % (lens[short_i], lens[long_i]))
print("  batched ALONE      -> first answer token pred=%r gold=%r hit=%s" % solo[0])
print("  batched WITH long  -> first answer token pred=%r gold=%r hit=%s" % paired[0])
print("  (the row itself is identical; only the batch padding changed)")

# AR first token on the same row
batch = collate([records[short_i]])
ids = {k: v.to(device) for k, v in batch.items() if k != "records"}
with torch.no_grad():
    prefix = model.encode_context_prefix(embed(ids["context_ids"]), ids["context_mask"])
    gen = model.generate_answer_with_prefix(prefix, ids["qa_context_indices"], ids["question_ids"],
                                            ids["question_mask"], tokenizer, 32)
print("  autoregressive     -> generated=%r" % tokenizer.decode(gen[0].tolist(), skip_special_tokens=True))

# ---------- BUG 2: max_new_tokens
print("=" * 90)
print("BUG 2: evaluation.max_new_tokens vs autoregressive F1")
for max_new in (32, 64, 128):
    t0 = time.time()
    proj, offi, lens_out = [], [], []
    for i in range(0, len(records), 8):
        chunk = records[i:i + 8]
        batch = collate(chunk)
        ids = {k: v.to(device) for k, v in batch.items() if k != "records"}
        with torch.no_grad():
            prefix = model.encode_context_prefix(embed(ids["context_ids"]), ids["context_mask"])
            for j in range(len(chunk)):
                g = model.generate_answer_with_prefix(prefix, ids["qa_context_indices"][j:j + 1],
                                                      ids["question_ids"][j:j + 1], ids["question_mask"][j:j + 1],
                                                      tokenizer, max_new)
                txt = tokenizer.decode(g[0].tolist(), skip_special_tokens=True)
                proj.append(example_metrics(txt, refs[i + j][:1])["f1"])
                offi.append(example_metrics(txt, refs[i + j])["f1"])
                lens_out.append(g.size(1))
        del prefix, ids, batch
        torch.cuda.empty_cache()
    print("max_new_tokens=%3d  project F1=%.4f official F1=%.4f  mean generated tokens=%.1f  %.0fs"
          % (max_new, sum(proj) / len(proj), sum(offi) / len(offi), sum(lens_out) / len(lens_out), time.time() - t0), flush=True)
