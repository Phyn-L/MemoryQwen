"""Check that the distributed evaluation reproduces the single-process metric.

Runs the same tiny validation set through Evaluator.teacher_forced / .autoregressive
with the loader sharded across `world_size` ranks (accelerator.prepare) and the
accumulators all-reduced.  If the reduction is correct, the numbers must match a
--num_processes 1 run bit for bit (up to float64 summation order).

Usage:
  accelerate launch --num_processes 1 ddp_eval_check.py <config> <tag>
  accelerate launch --num_processes 2 ddp_eval_check.py <config> <tag>
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path("/data/lz/MemoryQwen")
sys.path.insert(0, str(REPO))

from accelerate import Accelerator  # noqa: E402
from src.data import AggregatedContextDataset, collate_fn  # noqa: E402
from src.evaluator import Evaluator  # noqa: E402
from src.model import load_model  # noqa: E402
from utils.config import TrainConfig  # noqa: E402

cfg_path = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else "run"
cfg = TrainConfig.from_file(cfg_path)
cfg.validate()
# Identical initialisation across world sizes; without this the two runs are different
# random models and their metrics are not comparable.
random.seed(cfg.training.seed)
torch.manual_seed(cfg.training.seed)

accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="bf16")
tokenizer, model = load_model(cfg)
model.to(accelerator.device)

ds = AggregatedContextDataset(
    cfg.data.root, cfg.data.validation_datasets, cfg.data.validation_split, tokenizer,
    cfg.data.max_context_tokens, cfg.data.validation_max_samples,
    cfg.data.filter_long_context, cfg.data.filter_no_qa, allow_empty=True,
    cache_dir=Path("outputs/Qwen1.7B") if cfg.data.cache_dataset else None,
)
collate = lambda rows: collate_fn(
    rows, tokenizer,
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
loader = DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate)
model, loader = accelerator.prepare(model, loader)

evaluator = Evaluator(tokenizer, cfg)
with torch.no_grad():
    tf = evaluator.teacher_forced(accelerator.unwrap_model(model), loader, accelerator.device)
    ar = evaluator.autoregressive(
        accelerator.unwrap_model(model), loader, accelerator.device,
        include_teacher_metrics=False,
        max_qa=cfg.evaluation.autoregressive_max_qa,
    )

if accelerator.is_main_process:
    print("RESULT_JSON " + json.dumps({
        "tag": tag, "world_size": accelerator.num_processes,
        "tf_f1": round(tf["f1"], 6), "tf_em": round(tf["em"], 6),
        "tf_first_token_em": round(tf["first_token_em"], 6),
        "ar_f1": round(ar["f1"], 6), "ar_em": round(ar["em"], 6),
        "ar_first_token_em": round(ar["first_token_em"], 6),
    }, sort_keys=True), flush=True)
