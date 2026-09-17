from __future__ import annotations

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.evaluator import Evaluator
from src.model import load_model
from src.pipeline import make_collate, make_context_dataset
from utils.config import TrainConfig
from utils.machines import machine_names
from utils.checkpoint import CheckpointManager


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Qwen memory checkpoint")
    parser.add_argument("--config", default="configs/qwen-1.7b/train.yaml")
    parser.add_argument(
        "--machine", choices=machine_names(), default=None,
        help="Machine whose paths (utils/machines.py) this run uses.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--batch-size", type=int,
        help="Contexts per batch for the evaluation loader (default: the checkpoint config's "
             "training.batch_size). Evaluation runs under torch.no_grad(), so this is not "
             "bounded by the training memory -- but every QA pair of a batch's contexts travels "
             "with it (sample_qa=False), so the QA rows and the [rows, tokens, vocab] logits "
             "grow with this number. It does not change the metrics.",
    )
    parser.add_argument(
        "--qa-batch-size", type=int,
        help="QA rows per autoregressive generation group and per teacher-forced loss chunk.",
    )
    parser.add_argument(
        "--allow-missing-trainable", action="store_true",
        help="Load a checkpoint that does not cover every trainable tensor (for example an "
             "older run without the context_lm head) instead of failing.",
    )
    args = parser.parse_args()
    cfg = TrainConfig.from_file(args.config, machine=args.machine); cfg.validate()
    for name, value in (("--batch-size", args.batch_size), ("--qa-batch-size", args.qa_batch_size)):
        if value is not None and value < 1:
            raise SystemExit(f"{name} must be >= 1, got {value}")
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.qa_batch_size is not None:
        cfg.evaluation.qa_batch_size = args.qa_batch_size
    checkpoint_path = Path(args.checkpoint).resolve()
    cfg.checkpoint.output_dir = str(checkpoint_path.parent)
    if not cfg.logging.wandb_run_name:
        cfg.logging.wandb_run_name = checkpoint_path.parent.name
    # Multi-GPU evaluation: `accelerate launch --num_processes N` (scripts/test.sh does it)
    # gives every rank one card, and only the *loader* is sharded here. The evaluator
    # all-reduces its own accumulators (src/evaluator.py::_distributed_sum), so sharding the
    # batches is all that is needed to divide the work.
    #
    # The model is deliberately NOT prepared: `accelerator.prepare(model)` would wrap it in DDP
    # and enable bf16 autocast, while the training-time evaluation scores the unwrapped model
    # outside autocast (it calls `accelerator.unwrap_model`) -- preparing it here would make
    # `scripts/test.py` report slightly different numbers than the run's own validation curve.
    accelerator = None
    try:
        from accelerate import Accelerator

        accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="no")
        device = accelerator.device
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model(cfg)
    CheckpointManager(cfg.checkpoint.output_dir).load(
        checkpoint_path, model, allow_missing_trainable=args.allow_missing_trainable,
    )
    model.to(device)
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
    if accelerator is not None:
        loader = accelerator.prepare(loader)
    is_main = accelerator is None or accelerator.is_main_process
    if not len(loader):
        if is_main:
            print(f"No usable records in {args.split}; skipping evaluation.")
        if accelerator is not None:
            # Every rank must still take part in nothing else; just leave together.
            accelerator.wait_for_everyone()
        return
    if is_main:
        ranks = 1 if accelerator is None else accelerator.num_processes
        print(f"evaluating {ds.__class__.__name__} split={args.split} "
              f"contexts/batch={cfg.training.batch_size} qa_group={cfg.evaluation.qa_batch_size} "
              f"batches/rank={len(loader)} ranks={ranks}")
    evaluator = Evaluator(tokenizer, cfg)
    # Autoregressive first: it is the headline result. Teacher forcing feeds the answer
    # prefix back in, so its em/f1 mostly rewards lexical continuation; only
    # first_token_em there is a retrieval signal. Both collectives are called by every rank;
    # the numbers that come back are already the global ones.
    autoregressive = evaluator.autoregressive(model, loader, device, include_teacher_metrics=False)
    teacher_forced = evaluator.teacher_forced(model, loader, device)
    if is_main:
        print("autoregressive (headline):", autoregressive)
        print("teacher-forced (diagnostic only):", teacher_forced)
    if accelerator is not None:
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
