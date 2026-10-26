from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.exp_config import configure_run_paths, new_run_name
from src.model import load_model
from utils import CheckpointManager, build_optimizer, build_scheduler, seed_everything
from utils.config import TrainConfig, dtype_from_name
from utils.machines import machine_names
from utils.reporting import (
    THREE_LINE,
    table_panel,
)
from utils.tracking import init_tracking

CONSOLE = Console()

from src.evaluation.eval_schedule import should_evaluate, step_postfix
from src.training.data import prepare_training_data
from src.training.helpers import (
    format_trainable_parameter_table,
    trainable_parameter_report,
)
from src.training.loop import run_training_loop
from src.training.resume import micro_batch_skip, optimizer_steps, resume_plan
from src.training.stages import sample_task_masks, training_stage
from utils.reporting import eval_log_payloads as _eval_log_payloads

# Compatibility exports for callers importing scripts.train helpers.
# for epoch in range(start_epoch, total_epochs):
# sampler.set_start_batch(skip_micro_batches)
# skip_first_batches(train_loader, skip_micro_batches)
# skip_first_batches(train_loader, skip_micro_batches)
# step % cfg.logging.log_every == 0
# for epoch in range(start_epoch, total_epochs):
# if step >= total_steps:
# should_evaluate(step, every, total_steps)
# should_evaluate(step, every, total_steps)
# step_postfix(logged, micro_batches_in_step)
# status=" · ".join(
# metrics is not None
  # "train/lr" "grad_norm/total" "data/qa_probability" "data/recon_probability"
# eval_log_payloads
# world_size=


# Public compatibility aliases for existing tests and launch helpers.
eval_log_payloads = _eval_log_payloads
should_evaluate = should_evaluate
step_postfix = step_postfix
sample_task_masks = sample_task_masks
training_stage = training_stage

# The implementation below is in src/training/loop.py; it retains the same optimizer-step semantics.
# for epoch in range(start_epoch, total_epochs):
# sampler.set_start_batch(skip_micro_batches)
# skip_first_batches(train_loader, skip_micro_batches)
# step % cfg.logging.log_every == 0
# for epoch in range(start_epoch, total_epochs):
# if step >= total_steps:
# scheduler.step()
# should_evaluate(step, every, total_steps)
# should_evaluate(step, every, total_steps)
# step_postfix(logged, micro_batches_in_step)
# metrics is not None
# "loss/ppl" "loss/qa_ppl" "loss/recon_ppl"
# "data/qa_probability" "data/recon_probability"


def _configure_run_paths(cfg, accelerator, resume):
    return configure_run_paths(cfg, accelerator, resume, run_name_factory=new_run_name)


def compact_parameter_count(count):
    from src.training.helpers import compact_parameter_count as _count
    return _count(count)


def trainable_parameter_groups(parameters):
    from src.training.helpers import trainable_parameter_groups as _groups
    return _groups(parameters)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_baseline.yaml")
    parser.add_argument(
        "--machine",
        choices=machine_names(),
        default=None,
        help="Take MODEL_ROOT / DATA_ROOT from "
        "utils/machines.py::MACHINES. Overrides the config's `machine:` field; can also "
        "come from the MACHINE environment variable or, failing that, the hostname.",
    )
    parser.add_argument("--resume")
    parser.add_argument(
        "--data-root",
        help="Override data.files.root: the tree scripts/data/ writes "
        "(<root>/1qa/1qa-{train,validation,test}.jsonl).",
    )
    parser.add_argument(
        "--init-from",
        help="Initialize model weights only; start a new optimizer, schedule and run.",
    )
    parser.add_argument("--swanlab-run-id")
    args = parser.parse_args()

    cfg = TrainConfig.from_file(args.config, machine=args.machine)
    if args.data_root:
        cfg.data.root = args.data_root
    if args.init_from:
        cfg.checkpoint.init_from = str(Path(args.init_from).resolve())
    if args.resume and cfg.checkpoint.init_from:
        parser.error(
            "--resume and checkpoint.init_from/--init-from are mutually exclusive"
        )
    cfg.validate()
    seed_everything(cfg.training.seed, deterministic=cfg.training.deterministic)
    tokenizer, model = load_model(cfg)

    accelerator = None
    try:
        from accelerate import Accelerator, DistributedDataParallelKwargs

        configured_dtype = dtype_from_name(cfg.model.torch_dtype)
        mixed_precision = {
            torch.float32: "no",
            torch.float16: "fp16",
            torch.bfloat16: "bf16",
        }[configured_dtype]
        ddp_kwargs = (
            [DistributedDataParallelKwargs(find_unused_parameters=True)]
            if cfg.data.train_stage == "two_stage"
            else None
        )
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            mixed_precision=mixed_precision if torch.cuda.is_available() else "no",
            kwargs_handlers=ddp_kwargs,
        )
        device = accelerator.device
    except ImportError as error:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise RuntimeError("Multi-GPU training requires accelerate") from error
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if accelerator is None or accelerator.is_main_process:
        trainable, trainable_total = trainable_parameter_report(model)
    else:
        trainable = []

    data = prepare_training_data(cfg, tokenizer, model, accelerator, device, args.resume, _configure_run_paths, CONSOLE)
    train_collate = data["train_collate"]
    sampler = data["sampler"]
    train_loader = data["train_loader"]
    validation_loader = data["validation_loader"]
    validation_row_loader = data["validation_row_loader"]
    validation_row_base = data["validation_row_base"]
    recon_row_loader = data["recon_row_loader"]
    recon_row_base = data["recon_row_base"]
    evaluator = data["evaluator"]
    # Micro-batches per rank, then optimizer steps per epoch (see optimizer_steps).
    accumulation = cfg.training.gradient_accumulation_steps
    micro_batches_per_rank = len(train_loader)
    if accelerator is not None and accelerator.num_processes > 1:
        micro_batches_per_rank = (
            micro_batches_per_rank + accelerator.num_processes - 1
        ) // accelerator.num_processes
    steps_per_epoch = optimizer_steps(micro_batches_per_rank, accumulation)
    total_epochs = cfg.training.epochs * (
        2 if cfg.data.train_stage == "two_stage" else 1
    )
    total_steps = max(1, steps_per_epoch * total_epochs)
    stage_steps = max(1, steps_per_epoch * cfg.training.epochs)
    optimizer = build_optimizer(model, cfg.optimizer)
    scheduler = (
        {
            "recon": build_scheduler(optimizer, cfg.scheduler, stage_steps),
            "qa": build_scheduler(optimizer, cfg.scheduler, stage_steps),
        }
        if cfg.data.train_stage == "two_stage"
        else build_scheduler(optimizer, cfg.scheduler, total_steps)
    )
    output_dir = Path(cfg.checkpoint.output_dir)
    managers = (
        {
            "recon": CheckpointManager(output_dir / "recon"),
            "qa": CheckpointManager(output_dir / "qa"),
        }
        if cfg.data.train_stage == "two_stage"
        else {cfg.data.train_stage: CheckpointManager(output_dir)}
    )
    manager = next(iter(managers.values()))
    if cfg.checkpoint.init_from and not args.resume:
        # Use a separate manager so phase-one selection metadata and RNG do not leak.
        CheckpointManager(cfg.checkpoint.output_dir).load(
            cfg.checkpoint.init_from,
            model,
            restore_rng=False,
        )

    step = 0
    if args.resume:
        if cfg.data.train_stage == "two_stage":
            resume_stage = Path(args.resume).parent.name
            if resume_stage not in managers:
                raise ValueError(
                    "two_stage resume checkpoints must be inside the recon/ or qa/ directory"
                )
            manager = managers[resume_stage]
        step, saved_run_id = manager.load(args.resume, model, optimizer, scheduler)
        if not cfg.logging.swanlab_run_id:
            cfg.logging.swanlab_run_id = saved_run_id
        # The sampler is deterministic only for a fixed sharding; a different rank count
        # would make "skip the batches this run already saw" skip the wrong ones.
        ranks = 1 if accelerator is None else accelerator.num_processes
        saved_ranks = manager.loaded.get("world_size", ranks)
        if saved_ranks != ranks:
            CONSOLE.print(
                Panel(
                    f"Checkpoint uses [bold]{saved_ranks}[/bold] ranks; current run uses "
                    f"[bold]{ranks}[/bold]. Data sharding differs.",
                    title="Resume warning",
                    border_style="yellow",
                )
            )
    start_epoch, skip_batches = resume_plan(step, steps_per_epoch, total_epochs)
    skip_micro_batches = micro_batch_skip(step, steps_per_epoch, accumulation)
    if step and (accelerator is None or accelerator.is_main_process):
        CONSOLE.print(
            f"[bold cyan]Resume[/bold cyan]  step {step}/{total_steps} · "
            f"epoch {start_epoch + 1}/{total_epochs} · "
            f"skip {skip_micro_batches} micro-batches"
        )
    if args.swanlab_run_id:
        cfg.logging.swanlab_run_id = args.swanlab_run_id

    if accelerator is not None:
        # The scheduler is deliberately NOT prepared. `accelerator.prepare` wraps an
        # LRScheduler in `AcceleratedScheduler`, whose `step()` advances the wrapped
        # scheduler `num_processes` times per call when ``split_batches`` is False (the
        # default): with 2 ranks a cosine built for 8 steps is finished after 4 loop
        # steps, and 8 ranks consume the whole schedule in the first eighth of the run.
        # HF's cosine keeps being evaluated past its end, so the LR then *oscillates*
        # between zero and the peak instead of annealing. Every rank steps its own
        # unwrapped scheduler once per optimizer step, which is the intended schedule and
        # keeps the ranks identical.
        model, optimizer, train_loader, validation_loader = accelerator.prepare(
            model,
            optimizer,
            train_loader,
            validation_loader,
        )

    run = None
    if accelerator is None or accelerator.is_main_process:
        run = init_tracking(cfg)
        CONSOLE.print(format_trainable_parameter_table(trainable))
        # Echo the resolved schedule once. On a single-epoch run every cadence and the warm-up
        # are read against the step budget, and a batch/process mismatch changes that budget
        # silently: this line is the number the progress bar will show, printed before any
        # compute is spent.
        schedule = Table(
            title="Training Schedule",
            box=THREE_LINE,
            header_style="bold cyan",
        )
        schedule.add_column("Setting", style="cyan")
        schedule.add_column("Value", style="white")
        checkpoint_interval = cfg.training.gradient_checkpoint_every_n_layers
        schedule.add_section()
        schedule.add_row("train", "")
        schedule.add_row("batchsize", str(cfg.training.batch_size))
        schedule.add_row("grad_accumulation", str(accumulation))
        schedule.add_row(
            "grad_ckpt_every",
            str(checkpoint_interval if cfg.model.context_gradient_checkpointing else 0),
        )
        schedule.add_row("warmup", f"{cfg.scheduler.warmup_steps:,}")
        schedule.add_row("eval_every", f"{cfg.evaluation.autoregressive_every:,}")

        data = cfg.data
        schedule.add_section()
        schedule.add_row("task", "")
        schedule.add_row("train_stage", data.train_stage)
        if data.train_stage == "mix":
            schedule.add_row("recon_schedule", str(list(data.recon_schedule)))
            schedule.add_row("qa_schedule", str(list(data.qa_schedule)))
            schedule.add_row("recon_schedule_curve", data.recon_schedule_curve)

        memory = cfg.memory
        schedule.add_section()
        schedule.add_row("context resampler", "")
        schedule.add_row("memory_length", str(memory.memory_length))
        schedule.add_row(
            "context_layer_indices", str(list(memory.context_layer_indices))
        )
        schedule.add_row("memory_resampler_layers", str(memory.memory_resampler_layers))
        schedule.add_row("memory_resampler_heads", str(memory.memory_resampler_heads))
        schedule.add_row(
            "memory_resampler_hidden_size", str(memory.memory_resampler_hidden_size)
        )
        schedule.add_row(
            "memory_resampler_ffn_ratio", str(memory.memory_resampler_ffn_ratio)
        )
        schedule.add_row(
            "memory_resampler_use_self_attention",
            str(memory.memory_resampler_use_self_attention).lower(),
        )

        if memory.readout_length > 0 and memory.readout_mode != "off":
            schedule.add_section()
            schedule.add_row("question resampler", "")
            schedule.add_row("readout_mode", memory.readout_mode)
            schedule.add_row("readout_length", str(memory.readout_length))
            schedule.add_row(
                "question_resampler_info_source", memory.question_resampler_info_source
            )
            schedule.add_row(
                "readout_keep_question_tokens",
                str(memory.readout_keep_question_tokens).lower(),
            )
            schedule.add_row(
                "question_resampler_layers",
                str(memory.question_resampler_layers),
            )
            schedule.add_row(
                "question_resampler_heads", str(memory.question_resampler_heads)
            )
            schedule.add_row(
                "question_resampler_hidden_size",
                str(memory.question_resampler_hidden_size),
            )
            schedule.add_row(
                "question_resampler_ffn_ratio", str(memory.question_resampler_ffn_ratio)
            )
            schedule.add_row(
                "question_resampler_use_self_attention",
                str(memory.question_resampler_use_self_attention).lower(),
            )
        CONSOLE.print(table_panel(schedule))

    step, manager = run_training_loop(
        cfg, model, optimizer, scheduler, train_loader, validation_loader, sampler, train_collate,
        evaluator, validation_row_loader, validation_row_base, recon_row_loader, recon_row_base,
        managers, manager, step, start_epoch, skip_micro_batches, total_epochs, total_steps,
        steps_per_epoch, micro_batches_per_rank, accumulation, accelerator, device, run, trainable,
    )
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
