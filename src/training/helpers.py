from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
from rich.table import Table

from src.pipeline import model_run_label, new_run_name
from utils.config import TrainConfig
from utils.reporting import THREE_LINE, render_rich, table_panel


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

def _world_size(accelerator) -> int:
    """Ranks in this run; 1 when training without accelerate."""
    return 1 if accelerator is None else int(accelerator.num_processes)

def _process_index(accelerator) -> int:
    """This rank's index; 0 when training without accelerate."""
    return 0 if accelerator is None else int(accelerator.process_index)

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

def _configure_run_paths(cfg: TrainConfig, accelerator, resume: str | None, run_name_factory=new_run_name) -> None:
    if resume:
        run_dir = Path(resume).resolve().parent
        if cfg.data.train_stage == "two_stage" and run_dir.name in {"recon", "qa"}:
            run_dir = run_dir.parent
        cfg.checkpoint.output_dir = str(run_dir)
        cfg.logging.swanlab_run_name = cfg.logging.swanlab_run_name or run_dir.name
        cfg.logging.swanlab_log_dir = cfg.checkpoint.output_dir
        return
    is_main = accelerator is None or accelerator.is_main_process
    run_name = run_name_factory(cfg.model.name_or_path) if is_main else None
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
