"""Evaluate one ablation arm: teacher-forced + autoregressive SQuAD F1.

Usage: PYTHONPATH=<exp root> python eval_arm.py <config.yaml> <checkpoint.pt> [N]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parent
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO))

from src.data import ContextRecord, QARecord, collate_fn  # noqa: E402
from src.icl_baseline import example_metrics  # noqa: E402
from src.metrics import qa_metrics  # noqa: E402
from src.model import load_model  # noqa: E402
from utils.checkpoint import CheckpointManager  # noqa: E402
from utils.config import TrainConfig  # noqa: E402

cfg_path, ckpt = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 300
cfg = TrainConfig.from_file(cfg_path)
cfg.validate()
device = torch.device("cuda:0")
tokenizer, model = load_model(cfg)
step, _ = CheckpointManager(Path(ckpt).parent).load(ckpt, model)
model.to(device).eval()
embed = model.qwen.get_input_embeddings()

data = []
with open("/data/lz/contexts/aggregated/squad/validation.jsonl", encoding="utf-8") as fh:
    for line in fh:
        row = json.loads(line)
        for qa in row.get("qa_pairs", []) or []:
            ans = [str(a).strip() for a in (qa.get("answers") or []) if str(a).strip()]
            q = str(qa.get("question", "")).strip()
            if q and ans:
                data.append((ContextRecord(str(row["context"]),
                                           (QARecord(q, ans[0], "squad", ""),), "squad", ""), ans))
                break
        if len(data) >= N:
            break
records = [d[0] for d in data]
refs = [d[1] for d in data]


def collate(chunk):
    return collate_fn(chunk, tokenizer, cfg.data.max_context_tokens, cfg.data.max_question_tokens,
                      cfg.data.max_answer_tokens, cfg.data.append_eos, cfg.data.use_chat_template,
                      cfg.data.chat_template_enable_thinking, cfg.data.qa_per_context, False,
                      cfg.data.question_padding_side)


def report(tag, preds):
    proj = sum(qa_metrics(p, refs[i][0])["f1"] for i, p in enumerate(preds)) / len(preds)
    offi = sum(example_metrics(p, refs[i])["f1"] for i, p in enumerate(preds)) / len(preds)
    em = sum(example_metrics(p, refs[i])["em"] for i, p in enumerate(preds)) / len(preds)
    first = sum(float(bool(p.strip()) and bool(refs[i][0].strip())
                      and p.strip().split()[0].lower() == refs[i][0].strip().split()[0].lower())
                for i, p in enumerate(preds)) / len(preds)
    print("RESULT %s step=%d projectF1=%.4f officialF1=%.4f officialEM=%.4f firstWord=%.3f"
          % (tag, step, proj, offi, em, first), flush=True)
    return {"tag": tag, "step": step, "project_f1": proj, "official_f1": offi, "official_em": em, "first_word": first}


results = []
chunks = [records[i:i + 8] for i in range(0, len(records), 8)]
tf_preds, tf_first_hits = [], 0
for chunk in chunks:
    batch = collate(chunk)
    ids = {k: v.to(device) for k, v in batch.items() if k != "records"}
    with torch.no_grad():
        prefix = model.encode_context_prefix(embed(ids["context_ids"]), ids["context_mask"])
        out = model.forward_qa_with_prefix(
            prefix, ids["qa_context_indices"], embed(ids["question_ids"]), ids["question_mask"],
            embed(ids["answer_ids"]), ids["answer_mask"], ids["labels"])
    pred = out.logits.argmax(-1)[:, :-1]
    labels = out.labels[:, 1:]
    for i in range(len(chunk)):
        active = labels[i].ne(-100)
        tf_preds.append(tokenizer.decode(pred[i][active].tolist(), skip_special_tokens=True))
        tf_first_hits += int(pred[i][active][0].item() == labels[i][active][0].item())
    del out, prefix, ids, batch
    torch.cuda.empty_cache()
print("TF first-token exact accuracy = %.3f" % (tf_first_hits / len(records)), flush=True)
results.append(report("TF", tf_preds))

ar_preds = []
for chunk in chunks:
    batch = collate(chunk)
    ids = {k: v.to(device) for k, v in batch.items() if k != "records"}
    with torch.no_grad():
        prefix = model.encode_context_prefix(embed(ids["context_ids"]), ids["context_mask"])
        for j in range(len(chunk)):
            g = model.generate_answer_with_prefix(prefix, ids["qa_context_indices"][j:j + 1],
                                                  ids["question_ids"][j:j + 1], ids["question_mask"][j:j + 1],
                                                  tokenizer, cfg.evaluation.max_new_tokens)
            ar_preds.append(tokenizer.decode(g[0].tolist(), skip_special_tokens=True))
    del prefix, ids, batch
    torch.cuda.empty_cache()
results.append(report("AR", ar_preds))
Path(EXP / "results").mkdir(exist_ok=True)
(EXP / "results" / (Path(ckpt).parent.name + ".json")).write_text(
    json.dumps({"results": results, "predictions": {"tf": tf_preds, "ar": ar_preds}, "refs": refs}, ensure_ascii=False))
