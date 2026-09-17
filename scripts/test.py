from __future__ import annotations

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.evaluator import Evaluator
from src.model import load_model
from src.pipeline import make_collate, make_context_dataset
from utils.config import TrainConfig
from utils.checkpoint import CheckpointManager


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
    tokenizer, model = load_model(cfg)
    CheckpointManager(cfg.checkpoint.output_dir).load(
        checkpoint_path, model, allow_missing_trainable=args.allow_missing_trainable,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    ds = make_context_dataset(
        cfg, args.split, tokenizer,
        # --max-samples is the only cap here: the config's {split}_max_samples belongs to
        # the training loop, and silently applying it would change previous eval numbers.
        limit=args.max_samples,
        allow_empty=True,
    )
    loader = DataLoader(
        ds, batch_size=cfg.training.batch_size, shuffle=False,
        collate_fn=make_collate(cfg, tokenizer, sample_qa=False),
    )
    if not len(loader):
        print(f"No usable records in {args.split}; skipping evaluation.")
        return
    evaluator = Evaluator(tokenizer, cfg)
    # Autoregressive first: it is the headline result. Teacher forcing feeds the answer
    # prefix back in, so its em/f1 mostly rewards lexical continuation; only
    # first_token_em there is a retrieval signal.
    autoregressive = evaluator.autoregressive(model, loader, device, include_teacher_metrics=False)
    print("autoregressive (headline):", autoregressive)
    print("teacher-forced (diagnostic only):", evaluator.teacher_forced(model, loader, device))


if __name__ == "__main__":
    main()
