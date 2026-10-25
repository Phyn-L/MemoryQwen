from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data_pipeline import make_collate, make_context_dataset
from src.evaluation import Evaluator, evaluation_window_loader
from src.model import load_model
from utils.checkpoint import CheckpointManager
from utils.config import TrainConfig
from utils.machines import machine_names
from utils.model_paths import relocate_saved_paths
from utils.reporting import format_evaluation_report


def evaluation_config(checkpoint, config=None, machine=None):
    if config:
        raw_config = yaml.safe_load(Path(config).read_text())
        if "test" not in raw_config:
            return TrainConfig.from_file(config, machine=machine)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw = state.get("config")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Checkpoint has no saved config; supply --config explicitly")
    # The machine that wrote the checkpoint is provenance, not the current execution host.
    raw = dict(raw)
    raw.pop("machine", None)
    return TrainConfig.from_dict(
        relocate_saved_paths(raw, machine),
        machine=machine,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Qwen memory checkpoint"
    )
    parser.add_argument(
        "--config",
        help="Test YAML (checkpoint architecture), or explicit full training YAML",
    )
    parser.add_argument("--output-dir", help="Directory for evaluation JSON reports")
    parser.add_argument(
        "--report-file",
        help="Exact JSON report path (used by evaluation orchestration)",
    )
    parser.add_argument(
        "--machine",
        choices=machine_names(),
        default=None,
        help="Machine whose paths (utils/machines.py) this run uses.",
    )
    parser.add_argument("--checkpoint", "--ckpt", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--no-filter-long-context", action="store_true")
    parser.add_argument(
        "--data-file",
        help="Evaluate exactly this JSONL instead of the config's <split> file. Used by "
        "scripts/evaluation/test_all_suite.py to score one dataset at a time.",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--max-contexts-per-dataset",
        type=int,
        help="Maximum contexts evaluated per dataset (overrides the checkpoint config).",
    )
    parser.add_argument("--max-context-tokens", type=int)
    parser.add_argument(
        "--batch-size",
        "--bs",
        type=int,
        help="Contexts per batch for the evaluation loader (default: the checkpoint config's "
        "evaluation.ctx_batch_size). Evaluation runs under torch.no_grad(), so this is not "
        "bounded by the training memory -- but every QA pair of a batch's contexts travels "
        "with it (sample_qa=False), so the QA rows and the [rows, tokens, vocab] logits "
        "grow with this number. It does not change the metrics.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Upper bound on the tokens the autoregressive pass decodes per answer (default: "
        "evaluation.max_new_tokens from the checkpoint config, 32 in the shipped configs). "
        "The shipped evaluation configuration uses 32 tokens.",
    )
    parser.add_argument(
        "--qa-batch-size",
        type=int,
        help="QA rows per autoregressive generation group.",
    )
    parser.add_argument(
        "--allow-missing-trainable",
        action="store_true",
        help="Load a checkpoint that does not cover every trainable tensor instead of failing.",
    )
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    preliminary, _ = bootstrap.parse_known_args()
    if preliminary.config:
        raw = yaml.safe_load(Path(preliminary.config).read_text())
        if "test" in raw:
            if set(raw) - {"test", "machine"}:
                parser.error("Test YAML supports only test and machine sections")
            allowed = {
                "split",
                "ctx_batch_size",
                "qa_batch_size",
                "max_new_tokens",
                "max_samples",
            }
            if not isinstance(raw["test"], dict) or set(raw["test"]) - allowed:
                parser.error("Unknown test YAML fields")
            parser.set_defaults(**raw["test"], machine=raw.get("machine"))
    args = parser.parse_args()
    if args.split not in ("validation", "test"):
        parser.error("split must be validation or test")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("max-samples must be positive")
    cfg = evaluation_config(args.checkpoint, args.config, args.machine)
    if args.data_file:
        setattr(
            cfg.data,
            f"{args.split}_conversation_file",
            str(Path(args.data_file).expanduser().resolve()),
        )
    if args.max_context_tokens is not None:
        cfg.data.max_context_tokens = args.max_context_tokens
    for name, value in (
        ("--batch-size", args.batch_size),
        ("--qa-batch-size", args.qa_batch_size),
        ("--max-new-tokens", args.max_new_tokens),
        ("--max-contexts-per-dataset", args.max_contexts_per_dataset),
    ):
        if value is not None and value < 1:
            raise SystemExit(f"{name} must be >= 1, got {value}")
    if args.batch_size is not None:
        cfg.evaluation.ctx_batch_size = args.batch_size
    if args.qa_batch_size is not None:
        cfg.evaluation.qa_batch_size = args.qa_batch_size
    if args.max_new_tokens is not None:
        cfg.evaluation.max_new_tokens = args.max_new_tokens
    if args.max_contexts_per_dataset is not None:
        cfg.evaluation.max_eval_contexts_per_dataset = args.max_contexts_per_dataset
    if args.no_filter_long_context:
        cfg.data.filter_long_context = False
    cfg.validate()
    # Evaluation answers must fit the decoder budget; training keeps its own answer limit.
    cfg.data.max_answer_tokens = cfg.evaluation.max_new_tokens
    cfg.data.filter_qa_pair_answers = True
    checkpoint_path = Path(args.checkpoint).resolve()
    cfg.checkpoint.output_dir = str(checkpoint_path.parent)
    if not cfg.logging.swanlab_run_name:
        cfg.logging.swanlab_run_name = checkpoint_path.parent.name
    # Multi-GPU evaluation: `torchrun` (utils.launcher starts one worker per GPU)
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
    except ImportError as error:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise RuntimeError("Multi-GPU evaluation requires accelerate") from error
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model(cfg)
    CheckpointManager(cfg.checkpoint.output_dir).load(
        checkpoint_path,
        model,
        allow_missing_trainable=args.allow_missing_trainable,
    )
    model.to(device)
    ds = make_context_dataset(
        cfg,
        args.split,
        tokenizer,
        # --max-samples is the only cap here: the config's {split}_max_samples belongs to
        # the training loop, and silently applying it would change previous eval numbers.
        limit=args.max_samples,
        allow_empty=True,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.evaluation.ctx_batch_size,
        shuffle=False,
        collate_fn=make_collate(cfg, tokenizer, sample_qa=False),
        num_workers=cfg.data.eval_num_workers,
        pin_memory=cfg.data.eval_pin_memory,
        prefetch_factor=cfg.data.eval_prefetch_factor if cfg.data.eval_num_workers else None,
        persistent_workers=cfg.data.eval_persistent_workers and cfg.data.eval_num_workers > 0,
    )
    # Match training: sample QA turns once from the global seed, then reuse their unique
    # contexts for Recon. Each selected loader is partitioned across ranks.
    world = 1 if accelerator is None else int(accelerator.num_processes)
    rank = 0 if accelerator is None else int(accelerator.process_index)
    loader_options = dict(
        num_workers=cfg.data.eval_num_workers,
        pin_memory=cfg.data.eval_pin_memory,
        prefetch_factor=cfg.data.eval_prefetch_factor if cfg.data.eval_num_workers else None,
        persistent_workers=cfg.data.eval_persistent_workers and cfg.data.eval_num_workers > 0,
    )
    row_loader, row_base = evaluation_window_loader(
        ds,
        make_collate(cfg, tokenizer, sample_qa=False),
        budget=cfg.evaluation.max_eval_contexts_per_dataset,
        unit="rows",
        max_qa_sample_per_context=cfg.data.max_qa_sample_per_context,
        seed=cfg.training.seed,
        rank=rank,
        world=world,
        batch_size=cfg.evaluation.ctx_batch_size,
        max_new_tokens=cfg.evaluation.max_new_tokens,
        **loader_options,
    )
    recon_row_loader, recon_base = evaluation_window_loader(
        ds,
        make_collate(cfg, tokenizer, sample_qa=False),
        budget=cfg.evaluation.max_eval_contexts_per_dataset,
        unit="selected_contexts",
        selection=row_loader.evaluation_selection,
        max_qa_sample_per_context=cfg.data.max_qa_sample_per_context,
        rank=rank,
        world=world,
        batch_size=cfg.evaluation.ctx_batch_size,
        max_new_tokens=cfg.evaluation.max_new_tokens,
        **loader_options,
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
        print(
            f"evaluating {ds.__class__.__name__} split={args.split} "
            f"contexts/batch={cfg.evaluation.ctx_batch_size} qa_group={cfg.evaluation.qa_batch_size} "
            f"max_new_tokens={cfg.evaluation.max_new_tokens} "
            f"batches/rank={len(loader)} ranks={ranks}"
        )
    evaluator = Evaluator(tokenizer, cfg)
    import time

    started = time.time()
    if is_main:
        print("[eval] autoregressive pass (headline) ...", flush=True)
    task_metrics = evaluator.compare_reader_tasks(
        model,
        loader,
        device,
        max_contexts_per_dataset=cfg.evaluation.max_eval_contexts_per_dataset,
        max_qa_sample_per_context=cfg.data.max_qa_sample_per_context,
        row_loader=row_loader,
        row_base=row_base,
        recon_row_loader=recon_row_loader,
        recon_base=recon_base,
        pre_windowed=True,
    )
    if is_main:
        print(f"[eval] autoregressive done in {time.time() - started:.1f}s", flush=True)
    if is_main:
        print(
            format_evaluation_report(
                task_metrics,
                selection=row_loader.evaluation_selection,
                seed=cfg.training.seed,
                max_contexts=cfg.evaluation.max_eval_contexts_per_dataset,
                max_new_tokens=cfg.evaluation.max_new_tokens,
            )
        )
    if is_main:
        report = {
            "checkpoint": str(checkpoint_path),
            "split": args.split,
            "conversation_file": cfg.data.split_file(args.split),
            "config": cfg.to_dict(),
            **task_metrics,
        }
        if args.report_file:
            report_path = Path(args.report_file)
            report_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            from datetime import datetime

            report_dir = (
                Path(args.output_dir) if args.output_dir else checkpoint_path.parent
            )
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = (
                report_dir / f"eval_{args.split}_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
            )
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"results: {report_path}")
    if accelerator is not None:
        accelerator.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
