"""Evaluation facade preserving the training and CLI-facing Evaluator API."""

from __future__ import annotations

import torch
from rich.console import Console

from ..model import TASK_QA
from . import eval_distributed as _distributed
from .eval_aggregation import compare_reader_tasks as _compare_reader_tasks
from .eval_qa import (
    autoregressive as _autoregressive,
)
from .eval_qa import (
    first_token_hit,
    first_token_ids,
    token_text,
)
from .eval_recon import recon_teacher_forced as _recon_teacher_forced


def _make_eval_progress(total, description, rank, enabled, metric_name):
    """Facade for the progress helper; the console factory keeps tests injectable."""
    return _distributed.make_eval_progress(
        total,
        description,
        rank,
        enabled,
        metric_name,
        console_factory=Console,
    )


def _distributed_sum(values, device):
    return _distributed.distributed_sum(values, device)


def _distributed_world_size() -> int:
    return _distributed.distributed_world_size()


def _distributed_rank() -> int:
    return _distributed.distributed_rank()


def row_window(total: int, rank: int, world_size: int) -> tuple[int, int]:
    return _distributed.row_window(total, rank, world_size)


def _window_batch_count(window: tuple[int, int], batch_size: int | None) -> int | None:
    return _distributed.window_batch_count(window, batch_size)


class Evaluator:
    def __init__(self, tokenizer, cfg):
        self.tokenizer, self.cfg = tokenizer, cfg

    def _first_token_ids(self, record) -> set[int]:
        return first_token_ids(self.tokenizer, record)

    def _token_text(self, token_id: int) -> str:
        return token_text(self.tokenizer, token_id)

    def _first_token_hit(self, token_id: int, record) -> bool:
        return first_token_hit(self.tokenizer, token_id, record)

    @torch.no_grad()
    def autoregressive(
        self,
        model,
        loader,
        device,
        max_contexts_per_dataset=None,
        max_qa_sample_per_context=8,
        task_id=TASK_QA,
        row_loader=None,
        row_base=0,
        pre_windowed=False,
    ):
        return _autoregressive(
            self,
            model,
            loader,
            device,
            max_contexts_per_dataset=max_contexts_per_dataset,
            max_qa_sample_per_context=max_qa_sample_per_context,
            task_id=task_id,
            row_loader=row_loader,
            row_base=row_base,
            pre_windowed=pre_windowed,
        )

    def _evaluation_window(self, captioned, total, row_loader, loader, world, rank):
        return _distributed.evaluation_window(
            captioned, total, row_loader, loader, world, rank
        )

    @torch.no_grad()
    def recon_teacher_forced(
        self, model, loader, device, row_loader=None, context_base=0, pre_windowed=False
    ):
        return _recon_teacher_forced(
            self,
            model,
            loader,
            device,
            row_loader=row_loader,
            context_base=context_base,
            pre_windowed=pre_windowed,
        )

    @torch.no_grad()
    def compare_reader_tasks(
        self,
        model,
        loader,
        device,
        max_contexts_per_dataset=None,
        max_qa_sample_per_context=8,
        row_loader=None,
        include_recon=True,
        recon_loader=None,
        row_base=0,
        recon_row_loader=None,
        recon_base=0,
        pre_windowed=False,
    ):
        return _compare_reader_tasks(
            self,
            model,
            loader,
            device,
            max_contexts_per_dataset=max_contexts_per_dataset,
            max_qa_sample_per_context=max_qa_sample_per_context,
            row_loader=row_loader,
            include_recon=include_recon,
            recon_loader=recon_loader,
            row_base=row_base,
            recon_row_loader=recon_row_loader,
            recon_base=recon_base,
            pre_windowed=pre_windowed,
        )


__all__ = [
    "Evaluator",
    "_distributed_rank",
    "_distributed_sum",
    "_distributed_world_size",
    "_make_eval_progress",
    "_window_batch_count",
    "row_window",
]
