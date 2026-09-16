from __future__ import annotations

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.evaluator import Evaluator
from src.data import AggregatedContextDataset, collate_fn
from src.model import load_model
from utils.config import TrainConfig
from utils.checkpoint import CheckpointManager


def _model_label(model_path: str) -> str:
    import re
    match = re.search(r"Qwen(?:3)?[-_]?([0-9]+(?:\.[0-9]+)?)[Bb]", model_path, re.IGNORECASE)
    return f"Qwen{match.group(1)}B" if match else "Qwen"


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Qwen memory checkpoint")
    parser.add_argument("--config", default="configs/qwen-1.7b/train.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--allow-missing-trainable", action="store_true",
        help="Load a checkpoint that does not cover every trainable tensor (for example an "
             "older run without the context_lm head) instead of failing.",
    )
    args = parser.parse_args()
    cfg = TrainConfig.from_file(args.config); cfg.validate()
    checkpoint_path = Path(args.checkpoint).resolve()
    cfg.checkpoint.output_dir = str(checkpoint_path.parent)
    if not cfg.logging.wandb_run_name:
        cfg.logging.wandb_run_name = checkpoint_path.parent.name
    model_cache_dir = Path("outputs") / _model_label(cfg.model.name_or_path)
    tokenizer, model = load_model(cfg)
    CheckpointManager(cfg.checkpoint.output_dir).load(
        checkpoint_path, model, allow_missing_trainable=args.allow_missing_trainable,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    names = getattr(cfg.data, f"{args.split}_datasets") or cfg.data.dataset
    split_name = getattr(cfg.data, f"{args.split}_split")
    ds = AggregatedContextDataset(cfg.data.root, names, split_name, tokenizer, cfg.data.max_context_tokens, args.max_samples, cfg.data.filter_long_context, cfg.data.filter_no_qa, allow_empty=True, cache_dir=model_cache_dir if cfg.data.cache_dataset else None)
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
    loader = DataLoader(
        ds, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate,
    )
    if not len(loader):
        print(f"No usable records in {args.split}; skipping evaluation.")
        return
    evaluator = Evaluator(tokenizer, cfg)
    # Autoregressive first: it is the headline result. Teacher forcing feeds the gold
    # answer prefix back in, so its em/f1 mostly rewards lexical continuation; only
    # first_token_em there is a retrieval signal.
    autoregressive = evaluator.autoregressive(model, loader, device, include_teacher_metrics=False)
    print("autoregressive (headline):", autoregressive)
    print("teacher-forced (diagnostic only):", evaluator.teacher_forced(model, loader, device))


if __name__ == "__main__":
    main()
