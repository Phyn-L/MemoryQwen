"""Autoregressive QA evaluation and first-token diagnostics."""

from __future__ import annotations

import torch
import torch.distributed as dist

from utils.ddp import is_main_process

from ..model import TASK_QA
from .eval_distributed import (
    distributed_rank,
    distributed_sum,
    distributed_world_size,
    evaluation_window,
    make_eval_progress,
    window_batch_count,
)
from .eval_metrics import METRIC_KEYS, answer_line, best_reference_metrics, normalize_answer
from .eval_sampling import evaluation_window_loader


def first_token_ids(tokenizer, record) -> set[int]:
    first_ids = set()
    for reference in record.references:
        tokens = tokenizer(reference, add_special_tokens=False).input_ids
        if tokens:
            first_ids.add(int(tokens[0]))
    return first_ids


def token_text(tokenizer, token_id: int) -> str:
    return normalize_answer(
        tokenizer.decode([int(token_id)], skip_special_tokens=True)
    )


def first_token_hit(tokenizer, token_id: int, record) -> bool:
    produced = int(token_id)
    if produced == tokenizer.pad_token_id:
        return False
    produced_text = token_text(tokenizer, produced)
    for gold in first_token_ids(tokenizer, record):
        if produced == gold:
            return True
        if produced_text and produced_text == token_text(tokenizer, gold):
            return True
    return False


@torch.no_grad()
def autoregressive(
    evaluator,
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
    """Headline evaluation: every answer token is produced by the model itself."""
    was_training = model.training
    model.eval()
    sums = {key: 0.0 for key in METRIC_KEYS}
    dataset_sums = {}
    dataset_counts = {}
    samples = 0
    world, rank = distributed_world_size(), distributed_rank()
    enabled = is_main_process()
    if max_contexts_per_dataset is not None and not pre_windowed:
        source = loader if row_loader is None else row_loader
        row_loader, row_base = evaluation_window_loader(
            source.dataset,
            source.collate_fn,
            budget=max_contexts_per_dataset,
            unit="rows",
            max_qa_sample_per_context=max_qa_sample_per_context,
            rank=rank,
            world=world,
            batch_size=evaluator.cfg.evaluation.ctx_batch_size,
        )
        pre_windowed = True
    total = (
        getattr(row_loader, "evaluation_total", None)
        if max_contexts_per_dataset is not None
        else None
    )
    if max_contexts_per_dataset is not None and total is None:
        raise ValueError("Capped QA evaluation requires a per-dataset evaluation loader")
    if getattr(row_loader, "evaluation_prepartitioned", False):
        window, row_source = (0, None), row_loader
        cursor = 0
    else:
        window, row_source = evaluation_window(
            "evaluation.max_eval_contexts_per_dataset",
            total,
            row_loader,
            loader,
            world,
            rank,
        )
        cursor = int(row_base)
    task_name = "qa" if task_id == TASK_QA else "recon"
    desc = f"Validation (autoregressive, {task_name} task)"
    if world > 1:
        desc += f" [rank {rank}/{world}]"
    if pre_windowed or window[1] is None:
        local_batch_count = len(row_source)
    else:
        local_batch_count = window_batch_count(
            window, getattr(row_source, "batch_size", None)
        )
    progress, progress_task = make_eval_progress(
        local_batch_count, desc, rank, enabled, "qa_rows"
    )
    for batch in row_source:
        rows = batch["qa_input_ids"].size(0)
        batch_start, batch_end = cursor, cursor + rows
        cursor = batch_end
        start_row = max(batch_start, window[0])
        end_row = min(batch_end, batch_end if window[1] is None else window[1])
        if end_row <= start_row:
            if window[1] is not None and batch_start >= window[1]:
                break
            continue
        rows_slice = slice(start_row - batch_start, end_row - batch_start)
        ids = {
            key: value.to(device)
            for key, value in batch.items()
            if key != "records"
        }
        q = ids["qa_context_indices"][rows_slice]
        records = batch["records"][rows_slice]
        embed = model.qwen.get_input_embeddings()
        prefix = model.encode_context_prefix(
            embed(ids["context_ids"]), ids["context_mask"]
        )
        max_new_tokens = (
            evaluator.cfg.data.max_context_tokens
            if task_id != TASK_QA
            else evaluator.cfg.evaluation.max_new_tokens
        )
        if task_id != TASK_QA:
            question_ids = ids["qa_input_ids"].new_empty((end_row - start_row, 0))
            question_mask = ids["qa_input_mask"].new_empty(
                (end_row - start_row, 0), dtype=torch.bool
            )
            question_prefix_ids = ids["qa_prefix_ids"].new_empty((end_row - start_row, 0))
            question_prefix_mask = ids["qa_prefix_mask"].new_empty(
                (end_row - start_row, 0), dtype=torch.bool
            )
        else:
            question_ids = ids["qa_input_ids"][rows_slice]
            question_mask = ids["qa_input_mask"][rows_slice]
            question_prefix_ids = ids["qa_prefix_ids"][rows_slice]
            question_prefix_mask = ids["qa_prefix_mask"][rows_slice]
        generated = model.generate_answers_with_prefix(
            prefix,
            q,
            question_ids,
            question_mask,
            evaluator.tokenizer,
            max_new_tokens,
            group_by_context=True,
            max_rows_per_group=max(1, evaluator.cfg.evaluation.qa_batch_size),
            task_id=task_id,
            question_prefix_ids=question_prefix_ids,
            question_prefix_mask=question_prefix_mask,
            answer_prefix_ids=ids["qa_answer_prefix_ids"][rows_slice],
            answer_prefix_mask=ids["qa_answer_prefix_mask"][rows_slice],
            question_readout_mask=(
                ids["qa_readout_mask"][rows_slice] if task_id == TASK_QA else None
            ),
        )
        for i, record in enumerate(records):
            row = generated[i].tolist()
            prediction = answer_line(
                evaluator.tokenizer.decode(row, skip_special_tokens=True)
            )
            metrics = best_reference_metrics(prediction, record.references)
            for key, value in metrics.items():
                sums[key] += value
                dataset_sums.setdefault(
                    record.dataset, dict.fromkeys(METRIC_KEYS, 0.0)
                )[key] += value
            dataset_counts[record.dataset] = dataset_counts.get(record.dataset, 0) + 1
            samples += 1
        progress.update(progress_task, advance=1, qa_rows=f"QA rows {samples:,}")
        if window[1] is not None and cursor >= window[1]:
            break
    progress.stop()
    model.train(was_training)
    (samples,) = distributed_sum([samples], device)
    sums = dict(
        zip(
            METRIC_KEYS,
            distributed_sum([sums[key] for key in METRIC_KEYS], device),
            strict=True,
        )
    )
    result = {key: value / max(1, samples) for key, value in sums.items()}
    dataset_names = sorted(dataset_sums)
    if dist.is_available() and dist.is_initialized():
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, dataset_names)
        dataset_names = sorted({name for names in gathered for name in names})
    flat = []
    for name in dataset_names:
        flat.extend(dataset_sums.get(name, {}).get(key, 0.0) for key in METRIC_KEYS)
        flat.append(dataset_counts.get(name, 0))
    if flat:
        reduced = distributed_sum(flat, device)
        offset = 0
        for name in dataset_names:
            count = max(1.0, reduced[offset + len(METRIC_KEYS)])
            result[f"dataset/{name}/count"] = count
            for index, key in enumerate(METRIC_KEYS):
                result[f"dataset/{name}/{key}"] = reduced[offset + index] / count
            offset += len(METRIC_KEYS) + 1
    return result


__all__ = ["autoregressive", "first_token_hit", "first_token_ids", "token_text"]
