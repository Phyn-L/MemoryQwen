from __future__ import annotations

import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from utils.ddp import is_main_process

from .metrics import METRIC_KEYS, answer_line, best_reference_metrics, normalize_answer
from .icl_baseline import parse_race_prediction
from .model import TASK_QA, TASK_RECON


def _batch_value(batch, new_key, old_key):
    return batch[new_key] if new_key in batch else batch[old_key]


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
    tensor = torch.tensor([float(value) for value in values], dtype=torch.float64, device=device)
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
    per_rank = -(-int(total) // int(world_size))          # ceil
    start = min(int(total), int(rank) * per_rank)
    return start, min(int(total), start + per_rank)


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
    def autoregressive(self, model, loader, device, max_qa=None, task_id=TASK_QA,
                       row_loader=None):
        """Headline evaluation: every answer token is produced by the model itself.

        Metrics are computed from generated answers and gold references.

        ``max_qa`` caps how many QA rows the *whole* evaluation decodes: the budget is split
        into contiguous windows of the global row order (see :func:`row_window`), so the
        scored rows and the noise level stay the same when the rank count changes. Decoding
        is expensive, and a rank
        that stays inside the evaluation longer than the NCCL watchdog timeout (10 minutes)
        makes every other rank abort, which is why the budget exists at all.

        ``row_loader`` supplies that global order and must be the *unsharded* validation
        loader (``accelerator.prepare`` shards by batch, which would scatter a rank's rows
        across the whole split). Without it a distributed run cannot honour a global budget,
        so it falls back to the historical per-rank cap and says so.
        """
        was_training = model.training
        model.eval()
        sums = {key: 0.0 for key in METRIC_KEYS}
        dataset_sums = {}
        dataset_counts = {}
        race_correct = 0
        race_count = 0
        samples = 0
        first_token_hit = 0
        enabled = is_main_process()
        world, rank = _distributed_world_size(), _distributed_rank()
        if max_qa is None:
            window, row_source = (0, None), loader
        elif row_loader is None and world > 1:
            print(
                f"[evaluator] WARNING: evaluation.evaluation_max_samples={max_qa} is a global "
                f"row budget but no unsharded row loader was supplied; falling back to "
                f"{max_qa} rows per rank ({max_qa * world} rows total), which depends on the "
                "rank count.",
                flush=True,
            )
            window, row_source = (0, max_qa), loader
        else:
            window = row_window(max_qa, rank, world)
            row_source = loader if row_loader is None else row_loader
        cursor = 0
        task_name = "qa" if task_id == TASK_QA else "recon"
        desc = f"Validation (autoregressive, {task_name} task)"
        if world > 1:
            desc += f" [rank {rank}/{world}]"
        progress = tqdm(
            row_source,
            total=len(row_source),
            desc=desc,
            unit="batch",
            disable=not enabled,
            leave=False,
        )
        for batch in progress:
            rows = _batch_value(batch, "input_ids", "question_ids").size(0)
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
            if "input_ids" not in ids and "question_ids" in ids:
                ids["input_ids"], ids["input_mask"] = ids["question_ids"], ids["question_mask"]
            if "target_ids" not in ids and "answer_ids" in ids:
                ids["target_ids"], ids["target_mask"] = ids["answer_ids"], ids["answer_mask"]
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
            generated = model.generate_answers_with_prefix(
                prefix,
                q,
                ids["input_ids"][rows_slice],
                ids["input_mask"][rows_slice],
                self.tokenizer,
                self.cfg.evaluation.max_new_tokens,
                group_by_context=True,
                max_rows_per_group=max(1, self.cfg.evaluation.qa_batch_size),
                task_id=task_id,
            )
            for i, record in enumerate(records):
                row = generated[i].tolist()
                first_token_hit += int(
                    bool(row) and self._first_token_hit(row[0], record)
                )
                prediction = answer_line(self.tokenizer.decode(row, skip_special_tokens=True))
                metrics = best_reference_metrics(prediction, record.references)
                for key, value in metrics.items():
                    sums[key] += value
                    dataset_sums.setdefault(record.dataset, dict.fromkeys(METRIC_KEYS, 0.0))[key] += value
                dataset_counts[record.dataset] = dataset_counts.get(record.dataset, 0) + 1
                if record.dataset == "race":
                    _, predicted_letter = parse_race_prediction(prediction)
                    race_correct += int(predicted_letter == record.answer_letter)
                    race_count += 1
                samples += 1
            progress.set_postfix({"qa_rows": samples}, refresh=False)
            if window[1] is not None and cursor >= window[1]:
                break
        progress.close()
        model.train(was_training)
        first_token_hit, race_correct, race_count, samples = _distributed_sum([first_token_hit, race_correct, race_count, samples], device)
        sums = dict(
            zip(METRIC_KEYS, _distributed_sum([sums[key] for key in METRIC_KEYS], device))
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
        if race_count:
            result["accuracy"] = race_correct / race_count
        result["first_token_em"] = first_token_hit / max(1, samples)
        return result

    @torch.no_grad()
    def compare_task_embeddings(self, model, loader, device, max_qa=None, row_loader=None):
        """Evaluate identical QA rows with QA and Recon task embeddings."""
        return {
            "qa_task": self.autoregressive(
                model, loader, device, max_qa=max_qa, task_id=TASK_QA,
                row_loader=row_loader,
            ),
            "recon_task": self.autoregressive(
                model, loader, device, max_qa=max_qa, task_id=TASK_RECON,
                row_loader=row_loader,
            ),
        }
