"""Verify D1/D2 through the production Evaluator: teacher-forced F1 must not depend on
how many QA rows happen to share a batch (i.e. on how much question padding exists).

Legacy (right padding + overwrite EOS) is compared with the fixed
(left padding + append EOS) collate at batch sizes 1 and 8 contexts.

Usage: PYTHONPATH=. python .tmp_analysis/verify_fix2.py [N]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ContextRecord, QARecord, collate_fn  # noqa: E402
from src.evaluator import Evaluator  # noqa: E402
from src.model import load_model  # noqa: E402
from utils.checkpoint import CheckpointManager  # noqa: E402
from utils.config import TrainConfig  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
CKPT = sys.argv[2] if len(sys.argv) > 2 else "outputs/Qwen1.7B_20260916_024918/step-20000.pt"

cfg = TrainConfig.from_file("configs/qwen1.7b/train.yaml".replace("qwen1.7b", "qwen-1.7b"))
cfg.validate()
device = torch.device("cuda:0")
tokenizer, model = load_model(cfg)
step, _ = CheckpointManager(Path(CKPT).parent).load(CKPT, model)
model.to(device).eval()
print("ckpt step", step, flush=True)

rows = []
with open("/data/lz/contexts/aggregated/squad/validation.jsonl", encoding="utf-8") as fh:
    for line in fh:
        row = json.loads(line)
        pairs = []
        for qa in row.get("qa_pairs", []) or []:
            answers = [str(a).strip() for a in (qa.get("answers") or []) if str(a).strip()]
            question = str(qa.get("question", "")).strip()
            if question and answers:
                pairs.append(QARecord(question, answers[0], "squad", ""))
        if pairs:
            rows.append(ContextRecord(str(row["context"]), tuple(pairs), "squad", ""))
        if len(rows) >= N:
            break
print("contexts", len(rows), "qa rows", sum(len(r.qa_pairs) for r in rows), flush=True)


def loader_for(padding_side, eos_mode, batch_size):
    collate = lambda batch: collate_fn(
        batch, tokenizer,
        max_context_tokens=cfg.data.max_context_tokens,
        max_question_tokens=cfg.data.max_question_tokens,
        max_answer_tokens=cfg.data.max_answer_tokens,
        append_eos=cfg.data.append_eos,
        use_chat_template=cfg.data.use_chat_template,
        chat_template_enable_thinking=cfg.data.chat_template_enable_thinking,
        qa_per_context=cfg.data.qa_per_context,
        sample_qa=False,
        question_padding_side=padding_side,
        eos_mode=eos_mode,
    )
    return DataLoader(rows, batch_size=batch_size, shuffle=False, collate_fn=collate)


evaluator = Evaluator(tokenizer, cfg)
print("=" * 100)
with torch.no_grad():
    for padding_side, eos_mode, label in (
        ("right", "overwrite", "LEGACY (before fix)"),
        ("left", "append", "FIXED  (after fix) "),
    ):
        out = []
        for batch_size in (1, 8):
            r = evaluator.teacher_forced(model, loader_for(padding_side, eos_mode, batch_size), device)
            out.append(r)
            print("%s  batch=%d contexts -> TF f1=%.4f em=%.4f first_token_em=%.3f"
                  % (label, batch_size, r["f1"], r["em"], r["first_token_em"]), flush=True)
        print("%s  batch-composition sensitivity: dF1=%.4f  dEM=%.4f  dfirst_token=%.3f"
              % (label, out[1]["f1"] - out[0]["f1"], out[1]["em"] - out[0]["em"],
                 out[1]["first_token_em"] - out[0]["first_token_em"]), flush=True)
        print("-" * 100)
