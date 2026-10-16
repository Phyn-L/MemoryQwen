from __future__ import annotations

import math

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

from utils.ddp import is_main_process
from utils.progress import StepSpeedColumn

from .metrics import (
    METRIC_KEYS,
    answer_line,
    best_reference_metrics,
    normalize_answer,
)
from .model import TASK_QA, TASK_RECON


def _make_eval_progress(total, description, rank, enabled, metric_name):
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        StepSpeedColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn(f"[green]{{task.fields[{metric_name}]}}"),
        console=Console(stderr=True),
        # All ranks evaluate, but only rank zero may redraw the shared terminal.
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


def _distributed_sum(values, device):
    """All-reduce local scalar sums so that every rank sees the global total.

    ``accelerator.prepare`` shards the validation loader across ranks, so a metric built
    from one rank's batches only covers that shard. Every rank must run the evaluation
    and participate in this collective; reducing the accumulated sums (not the
    per-sample means) is what keeps the result equal to a single-process run.

    This is also why the evaluation must never be restricted to the main process: a rank
    that skips the call leaves the others waiting in an all-reduce until the NCCL
    watchdog aborts the job with "Watchdog caught collective operation timeout".
    """
    if not (dist.is_available() and dist.is_initialized()):
        return [float(value) for value in values]
    tensor = torch.tensor(
        [float(value) for value in values], dtype=torch.float64, device=device
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


def _distributed_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_world_size())
    return 1


def _distributed_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


def row_window(total: int, rank: int, world_size: int) -> tuple[int, int]:
    """The contiguous slice of the global QA-row order that one rank evaluates.

    The autoregressive budget is *global*: the windows of all ranks tile ``[0, total)``, so
    the union is the first ``total`` rows of the validation split no matter how many ranks
    there are. Giving every rank the same budget instead (the old behaviour) scored
    ``total * world_size`` rows whose identity changed with the rank count -- an 8-GPU run
    and a 4-GPU run were not comparable, and adding cards silently changed the test set.
    """
    if total <= 0 or world_size <= 1:
        return 0, max(0, total)
    per_rank = -(-int(total) // int(world_size))  # ceil
    start = min(int(total), int(rank) * per_rank)
    return start, min(int(total), start + per_rank)


def _window_batch_count(window: tuple[int, int], batch_size: int | None) -> int | None:
    """Number of batches intersecting a contiguous row window."""
    if batch_size is None or batch_size <= 0:
        return None
    start, end = window
    return max(0, -(-end // batch_size) - start // batch_size)


class Evaluator:
    def __init__(self, tokenizer, cfg):
        self.tokenizer, self.cfg = tokenizer, cfg

    def _first_token_ids(self, record) -> set[int]:
        """First token id of every gold answer of one QA pair.

        ``first_token_em`` asks whether the answer came from the memory alone, so it uses
        the same best-reference rule as the metrics: matching any annotator's first token
        counts as a hit.
        """
        first_ids = set()
        for reference in record.references:
            tokens = self.tokenizer(reference, add_special_tokens=False).input_ids
            if tokens:
                first_ids.add(int(tokens[0]))
        return first_ids

    def _token_text(self, token_id: int) -> str:
        """Normalized text of a single token; empty when it carries no content.

        Punctuation and special tokens normalize to the empty string, which is what keeps
        the text comparison in :meth:`_first_token_hit` from matching two of them.
        """
        return normalize_answer(
            self.tokenizer.decode([int(token_id)], skip_special_tokens=True)
        )

    def _first_token_hit(self, token_id: int, record) -> bool:
        """Whether one produced token is the gold first token of *any* reference.

        A token id comparison alone is not enough here. The same word has two encodings
        depending on the leading space: tokenizing the gold answer on its own yields
        ``cat``, while the model that generates after ``... Answer:`` emits `` cat``.
        Comparing the normalized text of the two single tokens closes that gap, and it is
        the alignment this diagnostic exists to measure -- a correct first word was being
        counted as a miss whenever the tokenizer placed a space in front of it.

        The comparison is not loosened: ids still match exactly, and the text fallback
        only applies when both sides normalize to something non-empty, so punctuation or
        a padding token cannot turn into a hit. Because the id comparison is tried first,
        this can only ever *add* hits: numbers reported before this change are lower
        bounds, not a different scale.
        """
        produced = int(token_id)
        if produced == self.tokenizer.pad_token_id:
            return False
        produced_text = self._token_text(produced)
        for gold in self._first_token_ids(record):
            if produced == gold:
                return True
            if produced_text and produced_text == self._token_text(gold):
                return True
        return False

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
        """Headline evaluation: every answer token is produced by the model itself.

        Metrics are computed from generated answers and gold references.

        ``max_contexts_per_dataset`` caps contexts; each contributes at most
        ``max_qa_sample_per_context`` QA turns. Contexts are balanced across ranks.
        """
        was_training = model.training
        model.eval()
        sums = {key: 0.0 for key in METRIC_KEYS}
        dataset_sums = {}
        dataset_counts = {}
        samples = 0
        world, rank = _distributed_world_size(), _distributed_rank()
        enabled = is_main_process()
        if max_contexts_per_dataset is not None and not pre_windowed:
            from .pipeline import evaluation_window_loader

            source = loader if row_loader is None else row_loader
            row_loader, row_base = evaluation_window_loader(
                source.dataset,
                source.collate_fn,
                budget=max_contexts_per_dataset,
                unit="rows",
                max_qa_sample_per_context=max_qa_sample_per_context,
                rank=rank,
                world=world,
                batch_size=source.batch_size,
            )
            pre_windowed = True
        total = (
            getattr(row_loader, "evaluation_total", None)
            if max_contexts_per_dataset is not None
            else None
        )
        if max_contexts_per_dataset is not None and total is None:
            raise ValueError(
                "Capped QA evaluation requires a per-dataset evaluation loader"
            )
        if getattr(row_loader, "evaluation_prepartitioned", False):
            window, row_source = (0, None), row_loader
            cursor = 0
        else:
            window, row_source = self._evaluation_window(
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
            local_batch_count = _window_batch_count(
                window, getattr(row_source, "batch_size", None)
            )
        progress, progress_task = _make_eval_progress(
            local_batch_count,
            desc,
            rank,
            enabled,
            "qa_rows",
        )
        for batch in row_source:
            rows = batch["qa_input_ids"].size(0)
            batch_start, batch_end = cursor, cursor + rows
            cursor = batch_end
            # Only the rows inside this rank's window are decoded. The batches outside it
            # are skipped *before* the prefix encode, so a rank whose window is late in the
            # split pays iteration cost only.
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
            # One batched call for the whole context batch: rows that share a context
            # share the same memory cache, so they are prefilled and decoded together
            # instead of one row at a time.  ``max_rows_per_group`` keeps the group
            # batch bounded by the same knob that used to bound the micro-batching.
            max_new_tokens = (
                self.cfg.data.max_context_tokens
                if task_id == TASK_RECON
                else self.cfg.evaluation.max_new_tokens
            )
            if task_id == TASK_RECON:
                question_ids = ids["qa_input_ids"].new_empty((end_row - start_row, 0))
                question_mask = ids["qa_input_mask"].new_empty(
                    (end_row - start_row, 0), dtype=torch.bool
                )
            else:
                question_ids = ids["qa_input_ids"][rows_slice]
                question_mask = ids["qa_input_mask"][rows_slice]
            generated = model.generate_answers_with_prefix(
                prefix,
                q,
                question_ids,
                question_mask,
                self.tokenizer,
                max_new_tokens,
                group_by_context=True,
                max_rows_per_group=max(1, self.cfg.evaluation.qa_batch_size),
                task_id=task_id,
            )
            for i, record in enumerate(records):
                row = generated[i].tolist()
                prediction = answer_line(
                    self.tokenizer.decode(row, skip_special_tokens=True)
                )
                metrics = best_reference_metrics(prediction, record.references)
                for key, value in metrics.items():
                    sums[key] += value
                    dataset_sums.setdefault(
                        record.dataset, dict.fromkeys(METRIC_KEYS, 0.0)
                    )[key] += value
                dataset_counts[record.dataset] = (
                    dataset_counts.get(record.dataset, 0) + 1
                )
                samples += 1
            progress.update(progress_task, advance=1, qa_rows=f"QA rows {samples:,}")
            if window[1] is not None and cursor >= window[1]:
                break
        progress.stop()
        model.train(was_training)
        (samples,) = _distributed_sum([samples], device)
        sums = dict(
            zip(
                METRIC_KEYS,
                _distributed_sum([sums[key] for key in METRIC_KEYS], device),
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
            reduced = _distributed_sum(flat, device)
            offset = 0
            for name in dataset_names:
                count = max(1.0, reduced[offset + len(METRIC_KEYS)])
                result[f"dataset/{name}/count"] = count
                for index, key in enumerate(METRIC_KEYS):
                    result[f"dataset/{name}/{key}"] = reduced[offset + index] / count
                offset += len(METRIC_KEYS) + 1
        return result

    def _evaluation_window(self, captioned, total, row_loader, loader, world, rank):
        """Shared budget policy of the two evaluation passes (see :func:`row_window`)."""
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
        return row_window(total, rank, world), (
            loader if row_loader is None else row_loader
        )

    @torch.no_grad()
    def recon_teacher_forced(
        self, model, loader, device, row_loader=None, context_base=0, pre_windowed=False
    ):
        """Reconstruction score of the memory without decoding anything.

        Every validation context is reproduced behind ``[memory, recon_task_embedding]`` and
        scored token by token, exactly like the training objective's Recon term. One forward
        per batch instead of ``max_context_tokens`` greedy steps per context: that cost is what
        used to keep the Recon reader out of the training loop entirely, even though Recon is
        the objective that actually measures whether the memory carries the context.

        Reported overall and per dataset: teacher-forced cross entropy (``recon_ce``), its
        perplexity, and the share of context tokens reproduced exactly
        (``recon_token_accuracy``).
        """
        was_training = model.training
        model.eval()
        world, rank = _distributed_world_size(), _distributed_rank()
        enabled = is_main_process()
        window = (0, None)
        row_source = loader if row_loader is None else row_loader
        max_logits_rows = getattr(
            getattr(self.cfg, "loss", None), "max_logits_rows", 256
        )
        sums: dict[str, list[float]] = {}
        cursor = int(context_base)
        progress, progress_task = _make_eval_progress(
            (
                len(row_source)
                if pre_windowed or window[1] is None
                else _window_batch_count(
                    window, getattr(row_source, "batch_size", None)
                )
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
            # A context row is reported under the dataset of the QA turns that point at it.
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
        reduced = _distributed_sum(flat, device)
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
        """Evaluate a shared deterministic QA sample with QA and Recon readers."""
        if max_contexts_per_dataset is not None and row_loader is None:
            from .pipeline import evaluation_window_loader

            seed = getattr(getattr(self.cfg, "training", None), "seed", 0)
            world, rank = _distributed_world_size(), _distributed_rank()
            row_loader, row_base = evaluation_window_loader(
                loader.dataset,
                loader.collate_fn,
                budget=max_contexts_per_dataset,
                unit="rows",
                max_qa_sample_per_context=max_qa_sample_per_context,
                seed=seed,
                rank=rank,
                world=world,
                batch_size=loader.batch_size,
                max_new_tokens=getattr(
                    getattr(self.cfg, "evaluation", None),
                    "max_new_tokens",
                    64,
                ),
            )
            pre_windowed = True
        if (
            max_contexts_per_dataset is not None
            and row_loader is not None
            and recon_row_loader is None
            and hasattr(row_loader, "evaluation_selection")
        ):
            from .pipeline import evaluation_window_loader

            recon_row_loader, recon_base = evaluation_window_loader(
                row_loader.evaluation_source_dataset,
                row_loader.evaluation_collate,
                budget=max_contexts_per_dataset,
                unit="selected_contexts",
                selection=row_loader.evaluation_selection,
                rank=_distributed_rank(),
                world=_distributed_world_size(),
                batch_size=row_loader.batch_size,
                max_new_tokens=getattr(
                    getattr(self.cfg, "evaluation", None),
                    "max_new_tokens",
                    64,
                ),
                max_qa_sample_per_context=max_qa_sample_per_context,
            )
        metrics = {
            "qa_task": self.autoregressive(
                model,
                loader,
                device,
                max_contexts_per_dataset=max_contexts_per_dataset,
                max_qa_sample_per_context=max_qa_sample_per_context,
                task_id=TASK_QA,
                row_loader=row_loader,
                row_base=row_base,
                pre_windowed=pre_windowed,
            )
        }
        if not include_recon:
            return metrics
        settings = getattr(self.cfg, "evaluation", None)
        teacher_forced = getattr(settings, "recon_teacherforce", True)
        autoregressive = getattr(settings, "recon_autoregressive", False)
        if teacher_forced:
            metrics["recon_task"] = self.recon_teacher_forced(
                model,
                loader if recon_loader is None else recon_loader,
                device,
                row_loader=row_loader if recon_row_loader is None else recon_row_loader,
                context_base=recon_base,
                pre_windowed=pre_windowed,
            )
        if autoregressive:
            metrics["recon_autoregressive"] = self.autoregressive(
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
