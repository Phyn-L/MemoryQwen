"""Distributed primitives shared by evaluation passes."""

from __future__ import annotations

import torch
import torch.distributed as dist
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from utils.progress import StepSpeedColumn


def make_eval_progress(total, description, rank, enabled, metric_name, console_factory=Console):
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        StepSpeedColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn(f"[green]{{task.fields[{metric_name}]}}"),
        console=console_factory(stderr=True),
        disable=not (enabled and rank == 0),
        transient=False,
        refresh_per_second=5,
    )
    task = progress.add_task(
        description,
        total=total,
        position=rank,
        speed_start_completed=0,
        **{metric_name: f"{metric_name} 0"},
    )
    progress.start()
    return progress, task


def distributed_sum(values, device):
    """All-reduce local scalar sums so every rank sees the global total."""
    if not (dist.is_available() and dist.is_initialized()):
        return [float(value) for value in values]
    tensor = torch.tensor(
        [float(value) for value in values], dtype=torch.float64, device=device
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


def distributed_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_world_size())
    return 1


def distributed_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


def row_window(total: int, rank: int, world_size: int) -> tuple[int, int]:
    """Return the contiguous global QA-row slice assigned to one rank."""
    if total <= 0 or world_size <= 1:
        return 0, max(0, total)
    per_rank = -(-int(total) // int(world_size))
    start = min(int(total), int(rank) * per_rank)
    return start, min(int(total), start + per_rank)


def window_batch_count(window: tuple[int, int], batch_size: int | None) -> int | None:
    """Return the number of batches intersecting a contiguous row window."""
    if batch_size is None or batch_size <= 0:
        return None
    start, end = window
    return max(0, -(-end // batch_size) - start // batch_size)


def evaluation_window(captioned, total, row_loader, loader, world, rank):
    """Apply the shared global-budget policy used by QA and Recon."""
    if total is None:
        return (0, None), loader
    if row_loader is None and world > 1:
        print(
            f"[evaluator] WARNING: {captioned} is a global budget but no unsharded row "
            f"loader was supplied; falling back to {total} rows per rank "
            f"({total * world} rows total), which depends on the rank count.",
            flush=True,
        )
        return (0, total), loader
    return row_window(total, rank, world), loader if row_loader is None else row_loader


__all__ = [
    "distributed_rank",
    "distributed_sum",
    "distributed_world_size",
    "evaluation_window",
    "make_eval_progress",
    "row_window",
    "window_batch_count",
]
