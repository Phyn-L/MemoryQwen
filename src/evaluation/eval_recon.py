"""Teacher-forced reconstruction evaluation."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from utils.ddp import is_main_process

from .eval_distributed import (
    distributed_rank,
    distributed_sum,
    distributed_world_size,
    make_eval_progress,
    window_batch_count,
)


@torch.no_grad()
def recon_teacher_forced(
    evaluator, model, loader, device, row_loader=None, context_base=0, pre_windowed=False
):
    """Score context reconstruction token by token without autoregressive decoding."""
    was_training = model.training
    model.eval()
    world, rank = distributed_world_size(), distributed_rank()
    enabled = is_main_process()
    window = (0, None)
    row_source = loader if row_loader is None else row_loader
    max_logits_rows = getattr(
        getattr(evaluator.cfg, "loss", None), "max_logits_rows", 256
    )
    cursor = int(context_base)
    sums: dict[str, list[float]] = {}
    progress, progress_task = make_eval_progress(
        (
            len(row_source)
            if pre_windowed or window[1] is None
            else window_batch_count(window, getattr(row_source, "batch_size", None))
        ),
        "Validation (teacher-forced reconstruction)"
        + (f" [rank {rank}/{world}]" if world > 1 else ""),
        rank,
        enabled,
        "contexts",
    )
    for batch in row_source:
        context_rows = batch["context_ids"].size(0)
        batch_start, batch_end = cursor, cursor + context_rows
        cursor = batch_end
        start_row = max(batch_start, window[0])
        end_row = min(batch_end, batch_end if window[1] is None else window[1])
        if end_row <= start_row:
            if window[1] is not None and batch_start >= window[1]:
                break
            continue
        records = batch["records"]
        ids = {
            key: value.to(device)
            for key, value in batch.items()
            if key != "records"
        }
        context_ids, context_mask = ids["context_ids"], ids["context_mask"]
        datasets = [""] * context_rows
        for turn, context_index in zip(
            records, ids["qa_context_indices"].tolist(), strict=True
        ):
            datasets[context_index] = turn.dataset
        embedding = model.qwen.get_input_embeddings()
        prefix = model.encode_context_prefix(embedding(context_ids), context_mask)
        stats = model.recon_teacher_forced(
            prefix,
            embedding(context_ids),
            context_mask,
            context_ids.masked_fill(~context_mask, -100),
            max_logits_rows=max_logits_rows,
        )
        nll, correct, tokens = (
            stats["nll"].tolist(),
            stats["correct"].tolist(),
            stats["tokens"].tolist(),
        )
        for row in range(start_row - batch_start, end_row - batch_start):
            bucket = sums.setdefault(
                datasets[row] or "unknown", [0.0, 0.0, 0.0, 0.0]
            )
            bucket[0] += nll[row]
            bucket[1] += correct[row]
            bucket[2] += tokens[row]
            bucket[3] += 1.0
        progress.update(
            progress_task,
            advance=1,
            contexts=f"Contexts {int(sum(v[3] for v in sums.values())):,}",
        )
        if window[1] is not None and cursor >= window[1]:
            break
    progress.stop()
    model.train(was_training)
    dataset_names = sorted(sums)
    if dist.is_available() and dist.is_initialized():
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, dataset_names)
        dataset_names = sorted({name for names in gathered for name in names})
    zero = [0.0, 0.0, 0.0, 0.0]
    flat: list[float] = []
    for name in dataset_names:
        flat.extend(sums.get(name, zero))
    flat.extend(
        [sum(sums.get(name, zero)[index] for name in sums) for index in range(4)]
    )
    reduced = distributed_sum(flat, device)
    result: dict[str, float] = {}
    for index, name in enumerate(dataset_names):
        nll, correct, tokens, _ = reduced[4 * index : 4 * index + 4]
        result[f"dataset/{name}/recon_ce"] = nll / max(tokens, 1.0)
        result[f"dataset/{name}/recon_token_accuracy"] = correct / max(tokens, 1.0)
        result[f"dataset/{name}/contexts"] = reduced[4 * index + 3]
    nll, correct, tokens, contexts = reduced[-4:]
    result["recon_ce"] = nll / max(tokens, 1.0)
    result["recon_ppl"] = math.exp(min(result["recon_ce"], 20.0))
    result["recon_token_accuracy"] = correct / max(tokens, 1.0)
    result["contexts"] = contexts
    return result


__all__ = ["recon_teacher_forced"]
