from __future__ import annotations

import argparse
import json
import yaml
from pathlib import Path
from utils.config_paths import resolve_config_path
import torch
from torch.utils.data import DataLoader

from src.evaluator import Evaluator
from src.model import load_model
from src.pipeline import make_collate, make_context_dataset
from utils.config import TrainConfig
from utils.machines import machine_names
from utils.checkpoint import CheckpointManager
from utils.model_paths import relocate_saved_paths


def evaluation_config(checkpoint, config=None, machine=None):
    if config:
        raw_config = yaml.safe_load(resolve_config_path(config).read_text())
        if 'test' not in raw_config:
            return TrainConfig.from_file(config, machine=machine)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw = state.get("config")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Checkpoint has no saved config; supply --config explicitly")
    # The machine that wrote the checkpoint is provenance, not the current execution host.
    raw = dict(raw)
    raw.pop("machine", None)
    return TrainConfig.from_dict(relocate_saved_paths(raw, machine), machine=machine)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Qwen memory checkpoint")
    parser.add_argument("--config", help="Test YAML (checkpoint architecture), or explicit full training YAML")
    parser.add_argument("--datasets", nargs="+", help="Dataset directory names under data.root")
    parser.add_argument(
        "--machine", choices=machine_names(), default=None,
        help="Machine whose paths (utils/machines.py) this run uses.",
    )
    parser.add_argument("--checkpoint", "--ckpt", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--source-version", choices=("ms_marco_v1_1", "ms_marco_v2_1"))
    parser.add_argument("--no-filter-long-context", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--batch-size", "--bs", type=int,
        help="Contexts per batch for the evaluation loader (default: the checkpoint config's "
             "training.batch_size). Evaluation runs under torch.no_grad(), so this is not "
             "bounded by the training memory -- but every QA pair of a batch's contexts travels "
             "with it (sample_qa=False), so the QA rows and the [rows, tokens, vocab] logits "
             "grow with this number. It does not change the metrics.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int,
        help="Upper bound on the tokens the autoregressive pass decodes per answer (default: "
             "evaluation.max_new_tokens from the checkpoint config, 32 in the shipped configs). "
             "Keep it at 32 to stay comparable with scripts/evaluation/test_icl_baseline.py, which uses "
             "--squad-max-new-tokens 32.",
    )
    parser.add_argument(
        "--qa-batch-size", type=int,
        help="QA rows per autoregressive generation group and per teacher-forced loss chunk.",
    )
    parser.add_argument(
        "--allow-missing-trainable", action="store_true",
        help="Load a checkpoint that does not cover every trainable tensor (for example an "
             "older run without the token_recon head) instead of failing.",
    )
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    preliminary, _ = bootstrap.parse_known_args()
    if preliminary.config:
        raw = yaml.safe_load(resolve_config_path(preliminary.config).read_text())
        if 'test' in raw:
            if set(raw) - {'test', 'machine'}:
                parser.error("Test YAML supports only test and machine sections")
            allowed = {'datasets', 'split', 'batch_size', 'qa_batch_size', 'max_new_tokens', 'max_samples'}
            if not isinstance(raw['test'], dict) or set(raw['test']) - allowed:
                parser.error("Unknown test YAML fields")
            parser.set_defaults(**raw['test'], machine=raw.get('machine'))
    args = parser.parse_args()
    if args.split not in ('validation', 'test'):
        parser.error("split must be validation or test")
    if args.datasets is not None and (not isinstance(args.datasets, (list, tuple)) or not args.datasets or
            any(not isinstance(x, str) or not x or '/' in x or x in ('.', '..') for x in args.datasets)):
        parser.error("datasets must be a list of dataset directory names")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("max-samples must be positive")
    cfg = evaluation_config(args.checkpoint, args.config, args.machine)
    if args.datasets:
        setattr(cfg.data, f"{args.split}_datasets", tuple(args.datasets))
    for name, value in (
        ("--batch-size", args.batch_size),
        ("--qa-batch-size", args.qa_batch_size),
        ("--max-new-tokens", args.max_new_tokens),
    ):
        if value is not None and value < 1:
            raise SystemExit(f"{name} must be >= 1, got {value}")
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.qa_batch_size is not None:
        cfg.evaluation.qa_batch_size = args.qa_batch_size
    if args.max_new_tokens is not None:
        cfg.evaluation.max_new_tokens = args.max_new_tokens
    if args.no_filter_long_context:
        cfg.data.filter_long_context = False
    cfg.validate()
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
        allow_empty=True, source_version=args.source_version,
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
              f"max_new_tokens={cfg.evaluation.max_new_tokens} "
              f"batches/rank={len(loader)} ranks={ranks}")
    evaluator = Evaluator(tokenizer, cfg)
    # Autoregressive first: it is the headline result. Teacher forcing feeds the answer
    # prefix back in, so its em/f1 mostly rewards lexical continuation; only
    # first_token_em there is a retrieval signal. Both collectives are called by every rank;
    # the numbers that come back are already the global ones.
    import time

    started = time.time()
    if is_main:
        print("[eval] autoregressive pass (headline) ...", flush=True)
    autoregressive = evaluator.autoregressive(model, loader, device, include_teacher_metrics=False)
    if is_main:
        print(f"[eval] autoregressive done in {time.time() - started:.1f}s", flush=True)
    started = time.time()
    if is_main:
        print("[eval] teacher-forced pass (diagnostic) ...", flush=True)
    teacher_forced = evaluator.teacher_forced(model, loader, device)
    if is_main:
        print(f"[eval] teacher-forced done in {time.time() - started:.1f}s", flush=True)
    if is_main:
        print("autoregressive (headline):", autoregressive)
        print("teacher-forced (diagnostic only):", teacher_forced)
    if is_main:
        report = {"checkpoint": str(checkpoint_path), "split": args.split,
                  "datasets": getattr(cfg.data, f"{args.split}_datasets") or cfg.data.dataset,
                  "config": cfg.to_dict(), "autoregressive": autoregressive,
                  "teacher_forced": teacher_forced}
        from datetime import datetime
        report_path = checkpoint_path.parent / f"eval_{args.split}_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"results: {report_path}")
    if accelerator is not None:
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
