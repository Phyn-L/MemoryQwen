"""Verify the D1/D2 fixes through the real Evaluator code path on a real checkpoint.

Runs evaluator.teacher_forced with the legacy (right padding / overwrite EOS) and the
fixed (left padding / append EOS) collate settings on the same 200 SQuAD-dev examples.

Usage: PYTHONPATH=. python .tmp_analysis/verify_fix.py [N]
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

cfg = TrainConfig.from_file("configs/qwen-1.7b/train.yaml")
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
        for qa in row.get("qa_pairs", []) or []:
            answers = [str(a).strip() for a in (qa.get("answers") or []) if str(a).strip()]
            question = str(qa.get("question", "")).strip()
            if question and answers:
                rows.append(ContextRecord(str(row["context"]),
                                          (QARecord(question, answers[0], "squad", ""),), "squad", ""))
                break
        if len(rows) >= N:
            break
print("examples", len(rows), flush=True)


def loader_for(padding_side, eos_mode):
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
    return DataLoader(rows, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate)


evaluator = Evaluator(tokenizer, cfg)
with torch.no_grad():
    for padding_side, eos_mode in (("right", "overwrite"), ("left", "append")):
        for split in ("validation", "test"):
            pass
        loader = loader_for(padding_side, eos_mode)
        tf = evaluator.teacher_forced(model, loader, device)
        ar = evaluator.autoregressive(model, loader, device, include_teacher_metrics=False)
        print("%-6s %-10s -> TF f1=%.4f em=%.4f first_token_em=%.3f | AR f1=%.4f em=%.4f first_token_em=%.3f"
              % (padding_side, eos_mode, tf["f1"], tf["em"], tf["first_token_em"],
                 ar["f1"], ar["em"], ar["first_token_em"]), flush=True)
