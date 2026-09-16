"""Oracle ablations on CPU.

Conditions, all on the same SQuAD-dev examples and the same trained checkpoint:
  A  current method: continuation sees only the 8 memory KV slots
  B  oracle: continuation sees the FULL [context + memory] KV cache (upper bound)
  C  base backbone in the identical raw prompt format (no memory, no LoRA) -> isolates
     "format/tokenisation" from "memory bottleneck"

Usage:
  /home/lz/miniconda3/envs/shine/bin/python .tmp_analysis/oracle.py [N] [MAX_CTX] [MAX_NEW]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ContextRecord, QARecord, collate_fn  # noqa: E402
from src.metrics import qa_metrics  # noqa: E402
from src.model import (  # noqa: E402
    ContextPrefix,
    build_block_causal_mask,
    build_continuation_mask,
    load_model,
)
from utils.checkpoint import CheckpointManager  # noqa: E402
from utils.config import TrainConfig  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
MAX_CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 768
MAX_NEW = int(sys.argv[3]) if len(sys.argv) > 3 else 24
CKPT = sys.argv[4] if len(sys.argv) > 4 else "outputs/Qwen1.7B_20260916_024918/step-20000.pt"

cfg = TrainConfig.from_file("configs/qwen-1.7b/train.yaml")
cfg.validate()
tokenizer, model = load_model(cfg)
step, _ = CheckpointManager(Path(CKPT).parent).load(CKPT, model)
model.eval()
embed = model.qwen.get_input_embeddings()

rows = []
with open("/data/lz/contexts/aggregated/squad/validation.jsonl", encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        pairs = []
        for qa in row.get("qa_pairs", []) or []:
            answers = qa.get("answers", qa.get("answer", []))
            answers = answers if isinstance(answers, list) else [answers]
            q = str(qa.get("question", "")).strip()
            if q and answers:
                pairs.append(QARecord(q, str(answers[0]).strip(), "squad", ""))
        if pairs:
            ids_ = tokenizer(str(row["context"]), add_special_tokens=False).input_ids[:MAX_CTX]
            rows.append(ContextRecord(tokenizer.decode(ids_, skip_special_tokens=True), (pairs[0],), "squad", ""))
        if len(rows) >= N:
            break
print(f"examples={len(rows)} max_ctx={MAX_CTX} ckpt={CKPT} step={step}", flush=True)

batch = collate_fn(
    rows, tokenizer, cfg.data.max_context_tokens, cfg.data.max_question_tokens,
    cfg.data.max_answer_tokens, cfg.data.append_eos, False, False,
    cfg.data.qa_per_context, False,
)
ids = {k: v for k, v in batch.items() if k != "records"}
context_embeds = embed(ids["context_ids"])
question_embeds = embed(ids["question_ids"])
answer_embeds = embed(ids["answer_ids"])

results = {k: [] for k in ("A_tf", "B_tf", "C_tf", "A_ar", "B_ar", "C_ar")}
details = []

with torch.no_grad():
    t0 = time.time()
    prefix = model.encode_context_prefix(context_embeds, ids["context_mask"])
    print("prefix encode %.1fs" % (time.time() - t0), flush=True)
    L, M = prefix.context_length, prefix.memory_length

    # ---------------- condition A: current method ----------------
    out = model.forward_qa_with_prefix(
        prefix, ids["qa_context_indices"], question_embeds, ids["question_mask"],
        answer_embeds, ids["answer_mask"], ids["labels"],
    )
    pred_a = out.logits.argmax(-1)[:, :-1]
    labels_a = out.labels[:, 1:]

    # ---------------- condition B: full context+memory KV visible ----------------
    full_cache = prefix.memory_cache  # rebuilt below with every prefix position
    layers = []
    # recompute full-cache prefix once (encode_context_prefix already discarded context KV)
    mem_inputs = model.memory_tokens.unsqueeze(0).expand(context_embeds.size(0), -1, -1)
    seq_pref = torch.cat([context_embeds, mem_inputs], dim=1)
    empty = ids["context_mask"].new_zeros(context_embeds.size(0), 0)
    pref_mask = build_block_causal_mask(ids["context_mask"], M, empty, empty, seq_pref.dtype)
    pref_out = model.qwen(
        inputs_embeds=seq_pref, attention_mask=pref_mask, use_cache=True,
        return_dict=True,
    )
    for layer in pref_out.past_key_values.layers:
        layers.append((layer.keys, layer.values))
    from transformers.cache_utils import DynamicCache
    full = DynamicCache(ddp_cache_data=layers, config=model.qwen.config)
    cur = torch.cat([question_embeds, answer_embeds], dim=1)
    full_mask = build_block_causal_mask(
        ids["context_mask"], M, ids["question_mask"], ids["answer_mask"], cur.dtype
    )[:, :, L + M:, :]
    positions = torch.arange(L + M, L + M + cur.size(1)).unsqueeze(0).expand(cur.size(0), -1)
    out_b = model.qwen(
        inputs_embeds=cur, attention_mask=full_mask, position_ids=positions,
        past_key_values=full, use_cache=True, return_dict=True,
    )
    ignored = torch.full((cur.size(0), question_embeds.size(1)), -100, dtype=ids["labels"].dtype)
    labels_b = torch.cat([ignored, ids["labels"]], dim=1)
    pred_b = out_b.logits.argmax(-1)[:, :-1]

    # ---------------- condition C: base backbone, no memory, no LoRA ----------------
    saved = [m.scaling for m in model.modules() if hasattr(m, "scaling") and hasattr(m, "lora_A")]
    for m in model.modules():
        if hasattr(m, "scaling") and hasattr(m, "lora_A"):
            m.scaling = 0.0
    seq_c = torch.cat([context_embeds, question_embeds, answer_embeds], dim=1)
    mask_c = build_block_causal_mask(
        ids["context_mask"], 0, ids["question_mask"], ids["answer_mask"], seq_c.dtype
    )
    pos_c = torch.arange(seq_c.size(1)).unsqueeze(0).expand(seq_c.size(0), -1)
    out_c = model.qwen(
        inputs_embeds=seq_c, attention_mask=mask_c, position_ids=pos_c,
        use_cache=False, return_dict=True,
    )
    qlen = question_embeds.size(1)
    pred_c = out_c.logits.argmax(-1)[:, :-1]
    labels_c = torch.cat(
        [torch.full((cur.size(0), L + qlen), -100, dtype=ids["labels"].dtype), ids["labels"]],
        dim=1,
    )[:, 1:]
    # restore LoRA
    for m, s in zip([m for m in model.modules() if hasattr(m, "scaling") and hasattr(m, "lora_A")], saved):
        m.scaling = s

    # ---------------- metrics ----------------
    first_tok = {"A": 0, "B": 0, "C": 0}
    for i, record in enumerate(batch["records"]):
        active = labels_a[i].ne(-100)
        tf_a = tokenizer.decode(pred_a[i][active].tolist(), skip_special_tokens=True)
        tf_b = tokenizer.decode(pred_b[i][active].tolist(), skip_special_tokens=True)
        active_c = labels_c[i].ne(-100)
        tf_c = tokenizer.decode(pred_c[i][active_c].tolist(), skip_special_tokens=True)
        # first answer token: the position where the memory must actually retrieve
        gold_first = ids["labels"][i][ids["labels"][i].ne(-100)][0].item()
        first_tok["A"] += int(pred_a[i][active][0].item() == gold_first)
        first_tok["B"] += int(pred_b[i][active][0].item() == gold_first)
        first_tok["C"] += int(pred_c[i][active_c][0].item() == gold_first)
        gen_a = model.generate_answer_with_prefix(
            prefix, ids["qa_context_indices"][i:i + 1], ids["question_ids"][i:i + 1],
            ids["question_mask"][i:i + 1], tokenizer, MAX_NEW,
        )
        ar_a = tokenizer.decode(gen_a[0].tolist(), skip_special_tokens=True)
        results["A_tf"].append(qa_metrics(tf_a, record.answer)["f1"])
        results["B_tf"].append(qa_metrics(tf_b, record.answer)["f1"])
        results["C_tf"].append(qa_metrics(tf_c, record.answer)["f1"])
        results["A_ar"].append(qa_metrics(ar_a, record.answer)["f1"])
        details.append((record.question, record.answer, tf_a, tf_b, tf_c, ar_a))
        print(f"[{i}] Q={record.question[:70]!r}\n    gold={record.answer!r}\n"
              f"    A_tf={tf_a!r}\n    B_tf={tf_b!r}\n    C_tf={tf_c!r}\n    A_ar={ar_a!r}", flush=True)

    # B autoregressive: reuse the full-cache continuation, then decode manually
    for i in range(cur.size(0)):
        cache = DynamicCache(
            ddp_cache_data=[(k[i:i + 1].clone(), v[i:i + 1].clone()) for k, v in layers],
            config=model.qwen.config,
        )
        q_valid = int(ids["question_mask"][i].sum())
        q_emb = question_embeds[i:i + 1, :q_valid]
        qm = ids["question_mask"][i:i + 1, :q_valid].bool()
        e = qm.new_zeros(1, 0)
        qmask = build_block_causal_mask(ids["context_mask"][i:i + 1], M, qm, e, q_emb.dtype)[:, :, L + M:, :]
        qpos = torch.arange(L + M, L + M + q_valid).unsqueeze(0)
        o = model.qwen(inputs_embeds=q_emb, attention_mask=qmask, position_ids=qpos,
                       past_key_values=cache, use_cache=True, return_dict=True)
        cache = o.past_key_values
        nxt = o.logits[:, -1].argmax(-1)
        gen = []
        for st in range(MAX_NEW):
            gen.append(int(nxt.item()))
            if int(nxt.item()) == tokenizer.eos_token_id:
                break
            tok = embed(nxt[:, None]).to(dtype=model.dtype)
            am = torch.ones((1, L + M + q_valid + st + 1), dtype=torch.bool)
            pid = torch.tensor([[L + M + q_valid + st]])
            o = model.qwen(inputs_embeds=tok, attention_mask=am, position_ids=pid,
                           past_key_values=cache, use_cache=True, return_dict=True)
            cache = o.past_key_values
            nxt = o.logits[:, -1].argmax(-1)
        ar_b = tokenizer.decode(gen, skip_special_tokens=True)
        results["B_ar"].append(qa_metrics(ar_b, details[i][1])["f1"])
        print(f"[{i}] B_ar={ar_b!r}", flush=True)

    # C autoregressive: base model, plain ids
    for i in range(cur.size(0)):
        c_ids = ids["context_ids"][i:i + 1]
        q_ids = ids["question_ids"][i:i + 1, :int(ids["question_mask"][i].sum())]
        full_ids = torch.cat([c_ids, q_ids], dim=1)
        am = torch.ones_like(full_ids, dtype=torch.bool)
        gen = model.qwen.generate(
            input_ids=full_ids, attention_mask=am, max_new_tokens=MAX_NEW,
            do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )
        ar_c = tokenizer.decode(gen[0, full_ids.size(1):].tolist(), skip_special_tokens=True)
        results["C_ar"].append(qa_metrics(ar_c, details[i][1])["f1"])
        print(f"[{i}] C_ar={ar_c!r}", flush=True)

print("=" * 80)
for key, values in results.items():
    print("%-6s n=%d  F1=%.4f" % (key, len(values), sum(values) / max(1, len(values))))
print("first-answer-token accuracy (n=%d): A(current)=%d B(full-KV)=%d C(base,no memory)=%d"
      % (len(batch["records"]), first_tok["A"], first_tok["B"], first_tok["C"]))
