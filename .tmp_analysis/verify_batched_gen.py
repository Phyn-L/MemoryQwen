"""Compare grouped batched generation against the original row-by-row path.

Both paths run on the same checkpoint, the same SQuAD-dev contexts and the same
greedy budget; only the batching of the decode loop differs.

Run: PYTHONPATH=. python .tmp_analysis/verify_batched_gen.py [N_CTX] [MAX_NEW]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/data/lz/MemoryQwen")

from src.data import ContextRecord, QARecord, collate_fn  # noqa: E402
from src.icl_baseline import example_metrics  # noqa: E402
from src.model import load_model  # noqa: E402
from utils.checkpoint import CheckpointManager  # noqa: E402
from utils.config import TrainConfig  # noqa: E402

N_CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 40
MAX_NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 32
CKPT = "outputs/Qwen1.7B_20260916_024918/step-20000.pt"

cfg = TrainConfig.from_file("configs/qwen-1.7b/train.yaml")
cfg.validate()
cfg.data.max_context_tokens = 2048
device = torch.device("cuda:0")
tokenizer, model = load_model(cfg)
step, _ = CheckpointManager(Path(CKPT).parent).load(CKPT, model)
model.to(device).eval()
print("checkpoint step", step, flush=True)

rows, refs = [], []
with open("/data/lz/contexts/aggregated/squad/validation.jsonl", encoding="utf-8") as fh:
    for line in fh:
        row = json.loads(line)
        pairs = []
        for qa in row.get("qa_pairs", []) or []:
            answers = [str(a).strip() for a in (qa.get("answers") or []) if str(a).strip()]
            question = str(qa.get("question", "")).strip()
            if question and answers:
                pairs.append((QARecord(question, answers[0], "squad", ""), answers))
        if pairs:
            # take every QA of the context so grouping actually has work to do
            rows.append(ContextRecord(str(row["context"]), tuple(p[0] for p in pairs), "squad", ""))
            refs.append([p[1] for p in pairs])
        if len(rows) >= N_CTX:
            break
records = [r for r in rows]
n_qa = sum(len(r.qa_pairs) for r in records)
print(f"contexts={len(records)} qa_rows={n_qa} max_new={MAX_NEW}", flush=True)

batch = collate_fn(
    records, tokenizer,
    max_context_tokens=cfg.data.max_context_tokens,
    max_question_tokens=cfg.data.max_question_tokens,
    max_answer_tokens=cfg.data.max_answer_tokens,
    append_eos=cfg.data.append_eos,
    use_chat_template=cfg.data.use_chat_template,
    chat_template_enable_thinking=cfg.data.chat_template_enable_thinking,
    qa_per_context=cfg.data.qa_per_context,
    sample_qa=False,
    question_padding_side=cfg.data.question_padding_side,
    eos_mode=cfg.data.eos_mode,
)
ids = {k: v.to(device) for k, v in batch.items() if k != "records"}
flat_refs = [ref for group in refs for ref in group]
assert len(flat_refs) == n_qa

with torch.no_grad():
    prefix = model.encode_context_prefix(
        model.qwen.get_input_embeddings()(ids["context_ids"]), ids["context_mask"]
    )
    outputs = {}
    for tag, grouped, per_group in (("row_by_row", False, 1), ("grouped", True, 8)):
        torch.cuda.synchronize()
        t0 = time.time()
        gen = model.generate_answers_with_prefix(
            prefix, ids["qa_context_indices"], ids["question_ids"], ids["question_mask"],
            tokenizer, MAX_NEW, group_by_context=grouped, max_rows_per_group=per_group,
        )
        torch.cuda.synchronize()
        texts = [tokenizer.decode(row.tolist(), skip_special_tokens=True) for row in gen]
        outputs[tag] = texts
        f1 = sum(example_metrics(texts[i], flat_refs[i])["f1"] for i in range(n_qa)) / n_qa
        em = sum(example_metrics(texts[i], flat_refs[i])["em"] for i in range(n_qa)) / n_qa
        print("%-11s %6.1fs   official F1=%.4f  EM=%.4f" % (tag, time.time() - t0, f1, em), flush=True)

import json as _json
_json.dump({"row_by_row": outputs["row_by_row"], "grouped": outputs["grouped"], "refs": flat_refs},
           open("/data/lz/MemoryQwen/.tmp_analysis/batched_gen_outputs.json", "w"), ensure_ascii=False)
a, b = outputs["row_by_row"], outputs["grouped"]
same = sum(1 for x, y in zip(a, b) if x == y)
print("identical outputs: %d/%d (%.1f%%)" % (same, n_qa, 100 * same / n_qa))
for i in range(n_qa):
    if a[i] != b[i]:
        print("  row %d\n    row_by_row=%r\n    grouped   =%r\n    gold      =%r"
              % (i, a[i], b[i], flat_refs[i][0]))
        if i > 0 and sum(1 for x, y in zip(a[:i], b[:i]) if x != y) >= 4:
            break
