"""Evaluation orchestration and shared QA/Recon aggregation."""

from __future__ import annotations

from ..model import TASK_QA, TASK_RECON
from .eval_distributed import distributed_rank, distributed_world_size
from .eval_sampling import evaluation_window_loader


def compare_reader_tasks(
    evaluator,
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
    qa_batch_sizes=None,
):
    """Evaluate one deterministic QA sample with both QA and Recon readers."""
    if max_contexts_per_dataset is not None and row_loader is None:
        seed = getattr(getattr(evaluator.cfg, "training", None), "seed", 0)
        world, rank = distributed_world_size(), distributed_rank()
        row_loader, row_base = evaluation_window_loader(
            loader.dataset,
            loader.collate_fn,
            budget=max_contexts_per_dataset,
            unit="rows",
            max_qa_sample_per_context=max_qa_sample_per_context,
            seed=seed,
            rank=rank,
            world=world,
            batch_size=evaluator.cfg.evaluation.ctx_batch_size,
            max_new_tokens=getattr(
                getattr(evaluator.cfg, "evaluation", None), "max_new_tokens", 64
            ),
        )
        pre_windowed = True
    if (
        max_contexts_per_dataset is not None
        and row_loader is not None
        and recon_row_loader is None
        and hasattr(row_loader, "evaluation_selection")
    ):
        recon_row_loader, recon_base = evaluation_window_loader(
            row_loader.evaluation_source_dataset,
            row_loader.evaluation_collate,
            budget=max_contexts_per_dataset,
            unit="selected_contexts",
            selection=row_loader.evaluation_selection,
            rank=distributed_rank(),
            world=distributed_world_size(),
            batch_size=evaluator.cfg.evaluation.ctx_batch_size,
            max_new_tokens=getattr(
                getattr(evaluator.cfg, "evaluation", None), "max_new_tokens", 64
            ),
            max_qa_sample_per_context=max_qa_sample_per_context,
        )
    metrics = {
        "qa_task": evaluator.autoregressive(
            model,
            loader,
            device,
            max_contexts_per_dataset=max_contexts_per_dataset,
            max_qa_sample_per_context=max_qa_sample_per_context,
            task_id=TASK_QA,
            row_loader=row_loader,
            row_base=row_base,
            pre_windowed=pre_windowed,
            qa_batch_sizes=qa_batch_sizes,
        )
    }
    if not include_recon:
        return metrics
    settings = getattr(evaluator.cfg, "evaluation", None)
    teacher_forced = getattr(settings, "recon_teacherforce", True)
    autoregressive_enabled = getattr(settings, "recon_autoregressive", False)
    if teacher_forced:
        metrics["recon_task"] = evaluator.recon_teacher_forced(
            model,
            loader if recon_loader is None else recon_loader,
            device,
            row_loader=row_loader if recon_row_loader is None else recon_row_loader,
            context_base=recon_base,
            pre_windowed=pre_windowed,
        )
    if autoregressive_enabled:
        metrics["recon_autoregressive"] = evaluator.autoregressive(
            model,
            loader,
            device,
            max_contexts_per_dataset=max_contexts_per_dataset,
            max_qa_sample_per_context=max_qa_sample_per_context,
            task_id=TASK_RECON,
            row_loader=row_loader,
            row_base=row_base,
            pre_windowed=pre_windowed,
        )
    return metrics


__all__ = ["compare_reader_tasks"]
