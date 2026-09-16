"""CPU diagnostic: inspect teacher-forced vs autoregressive outputs of a trained checkpoint.

Run with the 'shine' conda python from the repo root:
    /home/lz/miniconda3/envs/shine/bin/python .tmp_analysis/diag_ar.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ContextRecord, QARecord, collate_fn  # noqa: E402
from src.losses import qa_loss  # noqa: E402
from src.metrics import qa_metrics  # noqa: E402
from src.model import load_model  # noqa: E402
from utils.checkpoint import CheckpointManager  # noqa: E402
from utils.config import TrainConfig  # noqa: E402

N_EXAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 6
MAX_CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 512
MAX_NEW = int(sys.argv[3]) if len(sys.argv) > 3 else 24
CKPT = sys.argv[4] if len(sys.argv) > 4 else "outputs/Qwen1.7B_20260916_024918/step-20000.pt"

cfg = TrainConfig.from_file("configs/qwen-1.7b/train.yaml")
cfg.validate()

tokenizer, model = load_model(cfg)
step, _ = CheckpointManager(Path(CKPT).parent).load(CKPT, model)
model.eval()
print(f"loaded {CKPT} (step {step})", flush=True)

rows = []
with open("/data/lz/contexts/aggregated/squad/validation.jsonl", encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        pairs = []
        for qa in row.get("qa_pairs", []) or []:
            answers = qa.get("answers", qa.get("answer", []))
            answers = answers if isinstance(answers, list) else [answers]
            question = str(qa.get("question", "")).strip()
            if question and answers:
                pairs.append(QARecord(question, str(answers[0]).strip(), "squad", ""))
        if pairs:
            rows.append(ContextRecord(str(row["context"]), tuple(pairs), "squad", ""))
        if len(rows) >= N_EXAMPLES:
            break

# truncate contexts so CPU inference stays cheap
truncated = []
for record in rows:
    ids = tokenizer(record.context, add_special_tokens=False).input_ids[:MAX_CTX]
    truncated.append(
        ContextRecord(
            tokenizer.decode(ids, skip_special_tokens=True), record.qa_pairs[:1], "squad", ""
        )
    )

batch = collate_fn(
    truncated, tokenizer, cfg.data.max_context_tokens, cfg.data.max_question_tokens,
    cfg.data.max_answer_tokens, cfg.data.append_eos, cfg.data.use_chat_template,
    cfg.data.chat_template_enable_thinking, cfg.data.qa_per_context, False,
)
device = torch.device("cpu")
ids = {k: v for k, v in batch.items() if k != "records"}
embed = model.qwen.get_input_embeddings()

with torch.no_grad():
    prefix = model.encode_context_prefix(embed(ids["context_ids"]), ids["context_mask"])
    print(f"context tokens (padded)={prefix.context_length} memory={prefix.memory_length}", flush=True)
    out = model.forward_qa_with_prefix(
        prefix, ids["qa_context_indices"], embed(ids["question_ids"]), ids["question_mask"],
        embed(ids["answer_ids"]), ids["answer_mask"], ids["labels"],
    )
    print("qa_loss=%.4f" % float(qa_loss(out.logits, out.labels)), flush=True)
    predictions = out.logits.argmax(-1)[:, :-1]
    target = out.labels[:, 1:]
    for i, record in enumerate(batch["records"]):
        active = target[i].ne(-100)
        tf_text = tokenizer.decode(predictions[i][active].tolist(), skip_special_tokens=True)
        gold_text = tokenizer.decode(target[i][active].tolist(), skip_special_tokens=True)
        generated = model.generate_answer_with_prefix(
            prefix, ids["qa_context_indices"][i:i + 1], ids["question_ids"][i:i + 1],
            ids["question_mask"][i:i + 1], tokenizer, MAX_NEW,
        )
        ar_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        tf_m = qa_metrics(tf_text, record.answer)
        ar_m = qa_metrics(ar_text, record.answer)
        print("-" * 100)
        print("Q        :", record.question)
        print("gold     :", repr(record.answer), "| gold tok ids:", target[i][active].tolist())
        print("TF  pred :", repr(tf_text), "| ids:", predictions[i][active].tolist())
        print("TF  F1=%.3f EM=%.1f" % (tf_m["f1"], tf_m["em"]))
        print("AR  pred :", repr(ar_text), "| ids:", generated[0].tolist())
        print("AR  F1=%.3f EM=%.1f" % (ar_m["f1"], ar_m["em"]))
        print("gold text:", repr(gold_text), flush=True)
