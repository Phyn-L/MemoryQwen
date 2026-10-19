from __future__ import annotations

import argparse
import math
import os
from contextlib import nullcontext
from pathlib import Path

import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch.utils.data import DataLoader

from src.cache_build import DEFAULT_BUILD_WORKERS as BUILD_WORKERS
from src.cache_build import ensure_cache
from src.data import SortishSampler
from src.evaluator import Evaluator
from src.losses import objective_total, scheduled_task_weights, supervision_objectives
from src.model import load_model
from src.pipeline import (
    data_cache_dir,
    evaluation_window_loader,
    make_collate,
    make_context_dataset,
    model_run_label,
    new_run_name,
    resolve_split_limit,
)
from utils import CheckpointManager, build_optimizer, build_scheduler, seed_everything
from utils.config import TrainConfig, dtype_from_name
from utils.machines import machine_names
from utils.progress import StepSpeedColumn
from utils.reporting import (
    THREE_LINE,
    eval_log_payloads,
    format_evaluation_report,
    render_rich,
    table_panel,
)
from utils.tracking import init_tracking

CONSOLE = Console()


def resume_plan(step: int, steps_per_epoch: int, epochs: int) -> tuple[int, int]:
    """Where a resumed run picks up: ``(epoch to start at, batches to skip inside it)``.

    A checkpoint stores the optimizer/scheduler/step but not the position in the data
    stream, so before this a resumed run re-consumed the epoch from batch 0 *and* ran
    another full ``epochs`` pass on top of the loaded step, blowing past ``total_steps``.
    The sampler is deterministic for a fixed rank count, so the batches the run already
    saw can simply be skipped and the step budget stays authoritative.
    """
    if steps_per_epoch <= 0:
        return epochs, 0
    epoch = min(step // steps_per_epoch, epochs)
    if epoch >= epochs:
        return epochs, 0
    return epoch, step - epoch * steps_per_epoch


def optimizer_steps(micro_batches: int, accumulation: int) -> int:
    """Optimizer steps one epoch of ``micro_batches`` micro-batches produces.

    With accumulation the loop still consumes every batch, but only every ``accumulation``-th
    one closes an optimizer step (the scheduler, the evaluation cadence, the checkpoint step
    and ``total_steps`` all count optimizer steps).
    """
    if micro_batches <= 0:
        return 0
    return -(-int(micro_batches) // max(1, int(accumulation)))


def micro_batch_skip(step: int, steps_per_epoch: int, accumulation: int) -> int:
    """Micro-batches of the resume epoch that a run starting at optimizer ``step`` already saw.

    ``resume_plan`` reports the position in *optimizer* steps while the loop counts
    micro-batches; a checkpoint is only ever written on a group boundary, so the product is
    exact and the resumed run continues in the middle of no group.
    """
    if steps_per_epoch <= 0:
        return 0
    return (step % steps_per_epoch) * max(1, int(accumulation))


def trainable_parameter_report(
    model,
) -> tuple[list[tuple[str, tuple[int, ...], int]], int]:
    """Describe every trainable tensor; frozen LLM backbone weights are excluded."""
    parameters = [
        (name, tuple(parameter.shape), parameter.numel())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    return parameters, sum(count for _, _, count in parameters)


def trainable_parameter_groups(
    parameters: list[tuple[str, tuple[int, ...], int]],
) -> list[tuple[str, int, int]]:
    """Aggregate trainable tensors by their functional module."""
    groups: dict[str, list[int]] = {}
    for name, _, count in parameters:
        if "lora_A" in name or "lora_B" in name:
            group = "lora"
        elif name.startswith("context_resampler."):
            group = "context_resampler"
        elif name.startswith("resampler."):
            group = "question_resampler"
        elif name.startswith("context_layer_logits"):
            group = "context_layer_logits"
        elif name == "recon_task_embedding":
            group = "recon_task_embedding"
        else:
            group = name.split(".", 1)[0]
        entry = groups.setdefault(group, [0, 0])
        entry[0] += 1
        entry[1] += count
    return [
        (name, tensor_count, parameter_count)
        for name, (tensor_count, parameter_count) in sorted(groups.items())
    ]


def compact_parameter_count(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f} M"
    if count >= 1_000:
        return f"{count / 1_000:.1f} K"
    return f"{count:,}"


def format_trainable_parameter_table(
    parameters: list[tuple[str, tuple[int, ...], int]],
) -> str:
    type_names = {
        "lora": "LoRA adapters",
        "context_resampler": "ContextResampler",
        "question_resampler": "QuestionResampler",
        "context_layer_logits": "LayerWeights",
        "recon_task_embedding": "TaskEmbedding",
    }
    table = Table(
        box=THREE_LINE, header_style="bold cyan", title="Trainable Parameters"
    )
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="white")
    table.add_column("Params", justify="right", style="white")
    table.add_column("Tensors", justify="right", style="white")
    table.add_column("Mode", style="white")
    groups = trainable_parameter_groups(parameters)
    for index, (name, tensor_count, count) in enumerate(groups):
        table.add_row(
            str(index),
            name,
            type_names.get(name, "Parameter"),
            compact_parameter_count(count),
            str(tensor_count),
            "train",
        )
    table.add_row(
        "",
        "Trainable params",
        "",
        compact_parameter_count(sum(count for _, _, count in parameters)),
        str(sum(tensor_count for _, tensor_count, _ in groups)),
        "",
    )
    return render_rich(table_panel(table))


def should_evaluate(step: int, every: int, total_steps: int) -> bool:
    """Whether an evaluation runs at ``step``.

    The cadences are modulo-based, so a cadence that does not divide the run -- every 1000
    steps over a 7890-step single epoch, say -- would leave the last reported number 10%
    short of the end of training, and that final number is the one a short run is read
    for. The last step is therefore always evaluated, whatever the cadence says.
    """
    return step % every == 0 or step >= total_steps


def _world_size(accelerator) -> int:
    """Ranks in this run; 1 when training without accelerate."""
    return 1 if accelerator is None else int(accelerator.num_processes)


def _process_index(accelerator) -> int:
    """This rank's index; 0 when training without accelerate."""
    return 0 if accelerator is None else int(accelerator.process_index)


def step_postfix(
    terms: dict[str, float], micro_batches: int, max_ppl_exponent: float = 20.0
):
    """The tqdm fields for the accumulation group that is currently running.

    Values are sums over the group, so the display is the step's *mean* rather than the last
    micro-batch. It is one dict on purpose: ``set_postfix`` **replaces** the displayed fields,
    and a second call carrying only ``loss`` used to wipe the perplexity immediately after every
    optimizer step (with the default accumulation of 1, that was every step). It is returned as a
    plain dict in display order and passed positionally, because tqdm sorts keyword arguments.
    """
    means = {name: value / max(micro_batches, 1) for name, value in terms.items()}
    postfix = {"loss": f"{means.get('loss', 0.0):.4f}"}
    if "qa_loss" in means:
        postfix["qa"] = f"{means['qa_loss']:.4f}"
    if "recon_loss" in means:
        postfix["recon"] = f"{means['recon_loss']:.4f}"
    if "qa_loss" in means:
        # Same definition as `loss/ppl` in the SwanLab payload: perplexity of the QA objective.
        postfix["ppl"] = f"{math.exp(min(means['qa_loss'], max_ppl_exponent)):.2f}"
    return postfix


def accumulation_context(accelerator, model):
    """The context that turns a micro-batch into part of one optimizer step.

    ``accelerator.accumulate`` also drives the counter behind ``accelerator.sync_gradients``
    and wraps the model in ``no_sync`` so DDP all-reduces gradients once per group instead of
    once per micro-batch. Without accelerate there is nothing to defer: the loop divides the
    loss and steps the optimizer on the group boundary itself.
    """
    return accelerator.accumulate(model) if accelerator is not None else nullcontext()


def _gradient_norm(parameters) -> torch.Tensor:
    norms = [
        torch.linalg.vector_norm(parameter.grad.detach().float())
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack(norms))


def _configure_run_paths(cfg: TrainConfig, accelerator, resume: str | None) -> None:
    if resume:
        run_dir = Path(resume).resolve().parent
        cfg.checkpoint.output_dir = str(run_dir)
        cfg.logging.swanlab_run_name = cfg.logging.swanlab_run_name or run_dir.name
        cfg.logging.swanlab_log_dir = cfg.checkpoint.output_dir
        return
    is_main = accelerator is None or accelerator.is_main_process
    run_name = new_run_name(cfg.model.name_or_path) if is_main else None
    num_processes = accelerator.num_processes if accelerator is not None else 1
    if num_processes > 1:
        names = [run_name]
        torch.distributed.broadcast_object_list(names, src=0)
        run_name = names[0]
    cfg.logging.swanlab_run_name = cfg.logging.swanlab_run_name or run_name
    cfg.checkpoint.output_dir = str(
        Path(cfg.checkpoint.output_dir)
        / model_run_label(cfg.model.name_or_path)
        / run_name
    )
    cfg.logging.swanlab_log_dir = cfg.checkpoint.output_dir


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
        from accelerate import Accelerator

        configured_dtype = dtype_from_name(cfg.model.torch_dtype)
        mixed_precision = {
            torch.float32: "no",
            torch.float16: "fp16",
            torch.bfloat16: "bf16",
        }[configured_dtype]
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            mixed_precision=mixed_precision if torch.cuda.is_available() else "no",
        )
        device = accelerator.device
    except ImportError as error:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise RuntimeError("Multi-GPU training requires accelerate") from error
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if accelerator is None or accelerator.is_main_process:
        checkpoint_interval = cfg.training.gradient_checkpoint_every_n_layers
        checkpoint_depth = int(model.qwen.config.num_hidden_layers)
        checkpoint_state = (
            "Transformers enabled "
            f"every_n_layers={checkpoint_interval} "
            f"layers={len(range(0, checkpoint_depth, checkpoint_interval))}/{checkpoint_depth}"
            if cfg.model.context_gradient_checkpointing and checkpoint_interval > 0
            else "disabled"
        )
        CONSOLE.print(
            f"[bold cyan]Context checkpointing[/bold cyan]  {checkpoint_state}"
        )
        trainable, trainable_total = trainable_parameter_report(model)

    # The filtered cache (`cache/train.jsonl` + `cache/validation.jsonl`) is built from the
    # raw meta splits with this config's filters. Building it here keeps "which rows does this
    # run see" in one place; the work happens once (file-locked) no matter how many ranks start
    # together, and `scripts/data/build_cache.py` produces the same files by hand.
    if not cfg.data.train_conversation_file:
        built = ensure_cache(
            cfg.data,
            tokenizer,
            splits=("train", "validation"),
            quiet=accelerator is not None and not accelerator.is_main_process,
            # One pool for the whole run: the file lock means only one rank builds.
            num_workers=BUILD_WORKERS,
        )
        if accelerator is not None:
            accelerator.wait_for_everyone()
        if built and accelerator is not None and accelerator.is_main_process:
            CONSOLE.print(
                f"[bold yellow]Cache rebuilt[/bold yellow]  {', '.join(built)}"
            )

    train_file = Path(cfg.data.split_file("train"))
    validation_source = Path(cfg.data.validation_source())
    for path, what, build_with in (
        (train_file, "training", "train"),
        (validation_source, "validation", "validation"),
    ):
        if not path.is_file():
            hint = f"  python scripts/data/build_cache.py --config <this config> --splits {build_with}"
            if not Path(cfg.data.meta_file(build_with)).is_file():
                hint = f"  python scripts/data/build_1qa.py   # raw meta splits first\n{hint}"
            raise FileNotFoundError(
                f"{what} data not found: {path}\n"
                f"scripts/data/build_cache.py writes it from the raw meta split; run:\n{hint}"
            )
    model.to(device)
    if accelerator is not None:
        accelerator.wait_for_everyone()

    _configure_run_paths(cfg, accelerator, args.resume)
    cache_dir = data_cache_dir(cfg)

    train_dataset = make_context_dataset(
        cfg,
        "train",
        tokenizer,
        limit=resolve_split_limit(cfg, "train"),
        num_workers=cfg.data.train_num_workers,
    )
    train_collate = make_collate(cfg, tokenizer, sample_qa=True, training=True)
    validation_collate = make_collate(cfg, tokenizer, sample_qa=False)
    sampler = SortishSampler(
        train_dataset,
        tokenizer,
        cfg.training.batch_size,
        cfg.data.sortish_bucket_multiplier,
        cfg.training.seed,
        cfg.data.chat_template_enable_thinking,
        cache_dir if cfg.data.cache_sortish_lengths else None,
        # Deliberately no `num_workers` here, unlike the DataLoader below. This pass is one
        # batched tokenizer call over the contexts, which the tokenizer already parallelizes
        # internally: measured on 20k real contexts, 14224 context/s serial versus 4433 with 2
        # forked workers and 9985 with 8 (the batches are the whole context text, so pickling
        # them costs more than the tokenizing). It is cached after the first run either way.
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        sampler=sampler,
        collate_fn=train_collate,
        num_workers=cfg.data.train_num_workers,
        pin_memory=cfg.data.train_pin_memory,
        prefetch_factor=cfg.data.train_prefetch_factor if cfg.data.train_num_workers else None,
        persistent_workers=cfg.data.train_persistent_workers and cfg.data.train_num_workers > 0,
    )
    validation_dataset = make_context_dataset(
        cfg,
        "validation",
        tokenizer,
        limit=resolve_split_limit(cfg, "validation"),
        allow_empty=True,
        num_workers=cfg.data.eval_num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=validation_collate,
        num_workers=cfg.data.eval_num_workers,
        pin_memory=cfg.data.eval_pin_memory,
        prefetch_factor=cfg.data.eval_prefetch_factor if cfg.data.eval_num_workers else None,
        persistent_workers=cfg.data.eval_persistent_workers and cfg.data.eval_num_workers > 0,
    )
    # QA gets one seeded sample per dataset. Recon reuses the sample's unique contexts.
    # Both loaders are sharded manually, so accelerator.prepare must not touch them.
    loader_options = dict(
        num_workers=cfg.data.eval_num_workers,
        pin_memory=cfg.data.eval_pin_memory,
        prefetch_factor=cfg.data.eval_prefetch_factor if cfg.data.eval_num_workers else None,
        persistent_workers=cfg.data.eval_persistent_workers and cfg.data.eval_num_workers > 0,
    )
    validation_row_loader, validation_row_base = evaluation_window_loader(
        validation_dataset,
        validation_collate,
        budget=cfg.evaluation.max_eval_contexts_per_dataset,
        unit="rows",
        max_qa_sample_per_context=cfg.data.max_qa_sample_per_context,
        seed=cfg.training.seed,
        rank=_process_index(accelerator),
        world=_world_size(accelerator),
        batch_size=cfg.training.batch_size,
        max_new_tokens=cfg.evaluation.max_new_tokens,
        **loader_options,
    )
    recon_row_loader, recon_row_base = evaluation_window_loader(
        validation_dataset,
        validation_collate,
        budget=cfg.evaluation.max_eval_contexts_per_dataset,
        unit="selected_contexts",
        selection=validation_row_loader.evaluation_selection,
        max_qa_sample_per_context=cfg.data.max_qa_sample_per_context,
        rank=_process_index(accelerator),
        world=_world_size(accelerator),
        batch_size=cfg.training.batch_size,
        max_new_tokens=cfg.evaluation.max_new_tokens,
        **loader_options,
    )
    evaluator = Evaluator(tokenizer, cfg)
    # Micro-batches per rank, then optimizer steps per epoch (see optimizer_steps).
    accumulation = cfg.training.gradient_accumulation_steps
    micro_batches_per_rank = len(train_loader)
    if accelerator is not None and accelerator.num_processes > 1:
        micro_batches_per_rank = (
            micro_batches_per_rank + accelerator.num_processes - 1
        ) // accelerator.num_processes
    steps_per_epoch = optimizer_steps(micro_batches_per_rank, accumulation)
    total_steps = max(1, steps_per_epoch * cfg.training.epochs)
    optimizer = build_optimizer(model, cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg.scheduler, total_steps)
    manager = CheckpointManager(cfg.checkpoint.output_dir)
    if cfg.checkpoint.init_from and not args.resume:
        # Use a separate manager so phase-one selection metadata and RNG do not leak.
        CheckpointManager(cfg.checkpoint.output_dir).load(
            cfg.checkpoint.init_from,
            model,
            restore_rng=False,
        )

    step = 0
    if args.resume:
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
    start_epoch, skip_batches = resume_plan(step, steps_per_epoch, cfg.training.epochs)
    skip_micro_batches = micro_batch_skip(step, steps_per_epoch, accumulation)
    if step and (accelerator is None or accelerator.is_main_process):
        CONSOLE.print(
            f"[bold cyan]Resume[/bold cyan]  step {step}/{total_steps} · "
            f"epoch {start_epoch + 1}/{cfg.training.epochs} · "
            f"skip {skip_micro_batches} micro-batches"
        )
    finished = False
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
        ranks = 1 if accelerator is None else accelerator.num_processes
        schedule = Table(
            title="Training Schedule",
            box=THREE_LINE,
            header_style="bold cyan",
        )
        schedule.add_column("Setting", style="cyan")
        schedule.add_column("Value", style="white")
        schedule.add_column("Setting", style="cyan")
        schedule.add_column("Value", style="white")
        schedule.add_row(
            "Machine", cfg.machine or "config-defaults", "Steps", f"{total_steps:,}"
        )
        schedule.add_row("Epochs", str(cfg.training.epochs), "Ranks", str(ranks))
        schedule.add_row(
            "Batch / Rank",
            str(cfg.training.batch_size),
            "Accumulation",
            str(accumulation),
        )
        schedule.add_row(
            "Effective Batch",
            f"{cfg.training.batch_size * ranks * accumulation:,}",
            "Warmup",
            f"{cfg.scheduler.warmup_steps:,}",
        )
        schedule.add_row(
            "Eval Every",
            f"{cfg.evaluation.autoregressive_every:,}",
            "Log Every",
            f"{cfg.logging.log_every:,}",
        )
        CONSOLE.print(table_panel(schedule))

    model.train()
    # Deterministic objective sampling: the distillation position subset is drawn from a
    # generator reseeded with (training.seed, step), so it is recomputable from the step
    # number alone -- and therefore identical after a resume (see losses.sample_positions).
    sampling_generator = torch.Generator(device=device.type)
    # Scalar sums of the micro-batches in the group that is currently accumulating; the step
    # that closes the group logs and resets them.
    group_terms: dict[str, torch.Tensor] = {}
    group_micro_batches = 0
    progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        StepSpeedColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[green]{task.fields[status]}"),
        console=CONSOLE,
        disable=not (accelerator is None or accelerator.is_main_process),
    )
    progress_task = progress.add_task(
        "Training",
        total=total_steps,
        completed=step,
        status="",
        speed_start_completed=step,
    )
    progress.start()
    for epoch in range(start_epoch, cfg.training.epochs):
        sampler.set_epoch(epoch)
        sampler.set_start_batch(0)
        # Keep the deterministic QA-turn sampling of the collate on the same epoch.
        train_collate.set_epoch(epoch)
        epoch_loader = train_loader
        batch_offset = skip_micro_batches if epoch == start_epoch else 0
        if batch_offset:
            if accelerator is not None:
                # This wraps the prepared batch sampler, so skipped indices never reach
                # Dataset.__getitem__ or collate_fn on any rank.
                from accelerate.data_loader import skip_first_batches

                epoch_loader = skip_first_batches(train_loader, skip_micro_batches)
            else:
                sampler.set_start_batch(skip_micro_batches)
                epoch_loader = train_loader
        if hasattr(epoch_loader, "set_epoch"):
            epoch_loader.set_epoch(epoch)
        for batch_index, batch in enumerate(epoch_loader, start=batch_offset):
            if step >= total_steps:
                finished = True
                break
            # Accumulation: `accumulate` defers the DDP gradient all-reduce to the step
            # that closes the group (and tracks `sync_gradients`); without accelerate the
            # loop below does the same arithmetic by hand.
            sync_now = (
                batch_index + 1
            ) % accumulation == 0 or batch_index + 1 >= micro_batches_per_rank
            with accumulation_context(accelerator, model):
                sampling_generator.manual_seed(
                    cfg.training.seed
                    + step * accumulation
                    + (batch_index % accumulation)
                )
                ids = {
                    k: v.to(device, non_blocking=True)
                    for k, v in batch.items()
                    if k != "records"
                }
                base_model = (
                    accelerator.unwrap_model(model)
                    if accelerator is not None
                    else model
                )
                embedding = base_model.qwen.get_input_embeddings()
                recon_batch = {
                    "context_indices": ids["recon_context_indices"],
                    "context_embeds": embedding(ids["recon_target_ids"]),
                    "context_mask": ids["recon_target_mask"],
                    "labels": ids["recon_labels"],
                }
                output = model(
                    embedding(ids["context_ids"]),
                    ids["context_mask"],
                    embedding(ids["qa_input_ids"]),
                    ids["qa_input_mask"],
                    embedding(ids["qa_target_ids"]),
                    ids["qa_target_mask"],
                    ids["qa_labels"],
                    ids["qa_context_indices"],
                    context_ids=ids["context_ids"],
                    loss_config=cfg.loss,
                    loss_only=True,
                    recon_batch=recon_batch,
                    generator=sampling_generator,
                )
                task_ids = torch.cat(
                    [
                        torch.ones(
                            ids["qa_input_ids"].size(0), dtype=torch.long, device=device
                        ),
                        torch.zeros(
                            ids["recon_target_ids"].size(0),
                            dtype=torch.long,
                            device=device,
                        ),
                    ]
                )
                qa_weight, recon_weight, effective_ratio = scheduled_task_weights(
                    step, total_steps, cfg.data
                )
                supervised, task_terms = supervision_objectives(
                    output.supervision_losses,
                    task_ids,
                    cfg.loss,
                    qa_weight=qa_weight * cfg.loss.qa_weight,
                    recon_weight=recon_weight * cfg.loss.recon_weight,
                    row_mask=output.supervision_mask,
                )
                auxiliary_terms = output.objective_terms or {}
                total = supervised + objective_total(
                    supervised * 0, auxiliary_terms, cfg.loss
                )
                terms = {
                    **auxiliary_terms,
                    **task_terms,
                    "loss": total,
                    "qa_weight": total.detach() * 0 + qa_weight * cfg.loss.qa_weight,
                    "recon_weight": total.detach() * 0
                    + recon_weight * cfg.loss.recon_weight,
                    "recon_ratio": total.detach() * 0 + effective_ratio,
                }
                if accelerator is not None:
                    accelerator.backward(total)
                else:
                    # accelerate divides by gradient_accumulation_steps inside backward();
                    # the manual path has to do it here so the group is a true average.
                    (total / accumulation).backward()
                boundary = (
                    accelerator.sync_gradients if accelerator is not None else sync_now
                )
                for name, value in terms.items():
                    if isinstance(value, torch.Tensor):
                        detached = value.detach()
                        group_terms[name] = group_terms.get(name, detached.new_zeros(())) + detached
                group_micro_batches += 1
                if not boundary:
                    # Still accumulating: no optimizer step, no scheduler step, no evaluation.
                    continue
                log_step = (step + 1) % cfg.logging.log_every == 0
                if log_step:
                    named_parameters = list(base_model.named_parameters())
                    resampler_grad_norm = _gradient_norm(
                        parameter
                        for name, parameter in named_parameters
                        if name.startswith("context_resampler.")
                        and "layer_gates" not in name
                    )
                    metalora_grad_norm = _gradient_norm(
                        parameter
                        for name, parameter in named_parameters
                        if "lora_A" in name or "lora_B" in name
                    )
                    layer_gates_grad_norm = _gradient_norm(
                        parameter
                        for name, parameter in named_parameters
                        if name == "context_resampler.layer_gates"
                    )
                    recon_task_embedding_grad_norm = (
                        base_model.recon_task_embedding.grad.detach().float().norm()
                        if base_model.recon_task_embedding.grad is not None
                        else torch.zeros((), device=device)
                    )
                step_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.training.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                # Every logged scalar is the mean over the micro-batches of this step, which
                # is what the optimizer step actually optimised.
                logged = {
                    name: float(value / group_micro_batches)
                    for name, value in group_terms.items()
                }
                group_terms.clear()
                micro_batches_in_step = group_micro_batches
                group_micro_batches = 0

                postfix = step_postfix(logged, micro_batches_in_step)
                progress.update(
                    progress_task,
                    advance=1,
                    status=" · ".join(
                        [
                            *(f"{name} {value}" for name, value in postfix.items()),
                            f"lr {scheduler.get_last_lr()[0]:.2e}",
                        ]
                    ),
                )

                if run and log_step:
                    # `train/lr` is logged next to the losses on purpose: the LR schedule is
                    # the one training-side scalar that silently went wrong before (accelerate
                    # stepping it once per rank, see the prepare() comment), and without this
                    # key a distorted schedule is invisible in the run history.
                    payload = {
                        "loss/total": logged["loss"],
                        "loss/qa": logged["qa_loss"],
                        "loss/recon": logged["recon_loss"],
                        "loss/distill": logged["distill_loss"],
                        "train/lr": float(scheduler.get_last_lr()[0]),
                        "loss/ppl": float(math.exp(min(logged["qa_loss"], 20.0))),
                        "loss/qa_ppl": float(math.exp(min(logged["qa_loss"], 20.0))),
                        "loss/recon_ppl": float(
                            math.exp(min(logged["recon_loss"], 20.0))
                        ),
                        "grad_norm/total": float(step_grad_norm.detach()),
                        "grad_norm/resampler": float(resampler_grad_norm),
                        "grad_norm/metalora": float(metalora_grad_norm),
                        "grad_norm/layer_gates": float(layer_gates_grad_norm),
                        "grad_norm/recon_task_embedding": float(
                            recon_task_embedding_grad_norm
                        ),
                        "data/qa_probability": float(1.0 - effective_ratio),
                        "data/recon_probability": float(effective_ratio),
                        "data/micro_batches_per_step": float(micro_batches_in_step),
                    }
                    run.log(payload, step=step)
            is_main = accelerator is None or accelerator.is_main_process
            # Validation runs on EVERY rank.  accelerator.prepare shards the validation
            # loader, so evaluating only on the main process would (a) score one shard
            # instead of the whole split and (b) leave the other ranks waiting in the next
            # DDP collective for the entire evaluation, which the NCCL watchdog aborts
            # after 10 minutes ("Watchdog caught collective operation timeout").  The
            # evaluators all-reduce their accumulators, so every rank must call them.
            metrics = None
            if len(validation_loader) and should_evaluate(
                step, cfg.evaluation.autoregressive_every, total_steps
            ):
                # Eval uses its own Rich live display; leave only one display active.
                progress.stop()
                checkpoint_model = (
                    accelerator.unwrap_model(model)
                    if accelerator is not None
                    else model
                )
                step_checkpoint = None
                if is_main:
                    CONSOLE.log(f"[cyan]Saving checkpoint[/cyan]  step-{step:,}.pt")
                    step_checkpoint = manager.save(
                        checkpoint_model,
                        optimizer,
                        scheduler,
                        step,
                        cfg.to_dict(),
                        swanlab_run_id=getattr(run, "id", None),
                        world_size=(
                            1 if accelerator is None else accelerator.num_processes
                        ),
                    )
                    CONSOLE.log(
                        f"[green]Checkpoint saved[/green]  {step_checkpoint.name} · last.pt refreshed"
                    )
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                metrics = evaluator.compare_reader_tasks(
                    base_model,
                    validation_loader,
                    device,
                    max_contexts_per_dataset=cfg.evaluation.max_eval_contexts_per_dataset,
                    max_qa_sample_per_context=cfg.data.max_qa_sample_per_context,
                    row_loader=validation_row_loader,
                    row_base=validation_row_base,
                    recon_row_loader=recon_row_loader,
                    recon_base=recon_row_base,
                    pre_windowed=True,
                    include_recon=cfg.evaluation.recon_teacherforce
                    or cfg.evaluation.recon_autoregressive,
                )
                if is_main:
                    CONSOLE.print(
                        f"[bold green]Evaluation complete · step {step:,}[/bold green]"
                    )
                    CONSOLE.print(
                        format_evaluation_report(
                            metrics,
                            selection=validation_row_loader.evaluation_selection,
                            seed=cfg.training.seed,
                            step=step,
                            max_contexts=cfg.evaluation.max_eval_contexts_per_dataset,
                            max_new_tokens=cfg.evaluation.max_new_tokens,
                        )
                    )
            if run and is_main:
                for section, payload in eval_log_payloads(metrics):
                    for key in payload:
                        run.define_metric(key, section_name=section)
                    run.log(payload, step=step)
            if accelerator is not None:
                # Resynchronise explicitly so a rank whose shard finished early cannot
                # race ahead into the next training step.
                accelerator.wait_for_everyone()
            if metrics is not None and is_main:
                improved = manager.update_best(
                    step_checkpoint,
                    metrics["qa_task"]["f1"],
                )
                if improved:
                    CONSOLE.log(
                        f"[bold green]best.pt updated[/bold green]  step={step:,} · "
                        f"F1={metrics['qa_task']['f1']:.4f}"
                    )
                else:
                    if manager.best_f1 is None:
                        CONSOLE.log(
                            f"[yellow]best.pt unchanged[/yellow]  "
                            f"F1={metrics['qa_task']['f1']:.4f} is not finite"
                        )
                    else:
                        CONSOLE.log(
                            f"[dim]best.pt unchanged[/dim]  "
                            f"F1={metrics['qa_task']['f1']:.4f} · "
                            f"best F1={manager.best_f1:.4f} at step {manager.best_step:,}"
                        )
            if metrics is not None:
                progress.start()
        if finished:
            break

    if accelerator is None or accelerator.is_main_process:
        checkpoint_model = (
            accelerator.unwrap_model(model) if accelerator is not None else model
        )
        manager.save(
            checkpoint_model,
            optimizer,
            scheduler,
            step,
            cfg.to_dict(),
            final=True,
            swanlab_run_id=getattr(run, "id", None),
            world_size=1 if accelerator is None else accelerator.num_processes,
        )
    progress.stop()
    if run:
        run.finish()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
