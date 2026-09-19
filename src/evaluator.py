from __future__ import annotations

import math

import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from utils.ddp import is_main_process

from .losses import memory_objectives, objective_total, qa_loss
from .metrics import METRIC_KEYS, answer_line, best_reference_metrics, normalize_answer


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

    def _batch(self, model, prefix, batch, device, start=0, end=None):
        end = end or batch["question_ids"].size(0)
        ids = {
            key: value.to(device)
            for key, value in batch.items()
            if key not in {"records"}
        }
        qa_indices = ids["qa_context_indices"][start:end]
        embed = model.qwen.get_input_embeddings()
        output = model.forward_qa_with_prefix(
            prefix,
            qa_indices,
            embed(ids["question_ids"][start:end]),
            ids["question_mask"][start:end],
            embed(ids["answer_ids"][start:end]),
            ids["answer_mask"][start:end],
            ids["labels"][start:end],
        )
        qa = qa_loss(output.logits, output.labels)
        return output, qa, ids, qa_indices

    def _chunks(self, batch):
        size = self.cfg.evaluation.qa_batch_size
        return range(0, batch["question_ids"].size(0), size)

    def auxiliary_loss(self, model, prefix, ids):
        return memory_objectives(model, prefix, ids["context_ids"], ids["context_mask"], self.cfg.memory)

    @torch.no_grad()
    def teacher_forced(self, model, loader, device):
        was_training = model.training
        model.eval()
        qa_sum = 0.0
        auxiliary_sums = dict.fromkeys(("embedding_recon_loss", "token_recon_loss", "causal_recon_loss", "distill_loss"), 0.0)
        totals = {key: 0.0 for key in METRIC_KEYS}
        samples = 0
        first_token_hits = 0
        qa_weight = context_weight = 0
        enabled = is_main_process()
        ranks = _distributed_world_size()
        desc = "Validation (teacher-forced)"
        if ranks > 1:
            # Only the main rank draws a bar (four interleaved bars in one log are unreadable),
            # but the label has to say whose shard the ETA belongs to.
            desc += f" [rank {_distributed_rank()}/{ranks}]"
        progress = tqdm(
            loader,
            total=len(loader),
            desc=desc,
            unit="batch",
            disable=not enabled,
            leave=False,
        )
        for batch in progress:
            ids = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "records"
            }
            embed = model.qwen.get_input_embeddings()
            prefix = model.encode_context_prefix(
                embed(ids["context_ids"]), ids["context_mask"]
            )
            auxiliary = self.auxiliary_loss(model, prefix, ids)
            contexts = ids["context_ids"].size(0)
            for key, value in auxiliary.items():
                auxiliary_sums[key] += float(value) * contexts
            context_weight += contexts
            for start in self._chunks(batch):
                end = min(
                    start + self.cfg.evaluation.qa_batch_size,
                    batch["question_ids"].size(0),
                )
                output, qa, ids, _ = self._batch(
                    model, prefix, batch, device, start, end
                )
                count = end - start
                qa_sum += float(qa) * count
                qa_weight += count
                predictions = output.logits.argmax(-1)[:, :-1]
                target = output.labels[:, 1:]
                for i, record in enumerate(batch["records"][start:end]):
                    active = target[i].ne(-100)
                    if not bool(active.any()):
                        samples += 1
                        continue
                    # Retrieval diagnostic: the first answer token is the only
                    # answer position that must be produced from the memory alone
                    # (every later token can copy the teacher-forced prefix it was fed).
                    first_token_hits += int(
                        self._first_token_hit(
                            int(predictions[i][active][0].item()), record
                        )
                    )
                    text = self.tokenizer.decode(
                        predictions[i][active].tolist(), skip_special_tokens=True
                    )
                    # Same answer span the ICL baseline scores: its first line. See
                    # src.metrics.answer_line for why this is not cosmetic.
                    metrics = best_reference_metrics(answer_line(text), record.references)
                    for key, value in metrics.items():
                        totals[key] += value
                    samples += 1
            progress.set_postfix({"contexts": context_weight, "qa_rows": samples}, refresh=False)
        progress.close()
        model.train(was_training)
        qa_sum, first_token_hits, qa_weight, context_weight, samples = (
            _distributed_sum(
                [qa_sum, first_token_hits, qa_weight, context_weight, samples],
                device,
            )
        )
        metric_totals = dict(
            zip(METRIC_KEYS, _distributed_sum([totals[key] for key in METRIC_KEYS], device))
        )
        qa_value = qa_sum / max(1, qa_weight)
        auxiliary_values = dict(zip(auxiliary_sums, (value / max(1, context_weight) for value in _distributed_sum(list(auxiliary_sums.values()), device))))
        total_value = objective_total(qa_value, auxiliary_values, self.cfg.memory)
        return {
            "qa_loss": qa_value,
            "ppl": math.exp(min(qa_value, 20)),
            **auxiliary_values,
            "loss": total_value,
            **{key: value / max(1, samples) for key, value in metric_totals.items()},
            # Not an answer-quality score: teacher forcing feeds the gold answer
            # prefix, so every metric here mostly measures lexical continuation.
            # Treat first_token_em as the retrieval signal and prefer the
            # autoregressive numbers as the headline result.
            "first_token_em": first_token_hits / max(1, samples),
        }

    @torch.no_grad()
    def autoregressive(self, model, loader, device, include_teacher_metrics=True, max_qa=None,
                       row_loader=None):
        """Headline evaluation: every answer token is produced by the model itself.

        Unlike :meth:`teacher_forced` this never feeds the gold answer back, so these are
        the numbers to quote. The metrics use the single official SQuAD normalizer from
        ``src.metrics``, which is also what ``scripts/evaluation/test_icl_baseline.py`` scores with, so
        the two numbers are directly comparable. ``include_teacher_metrics`` additionally
        reports the teacher-forced ppl/qa_loss/recon_loss at the cost of a second
        pass.

        ``max_qa`` caps how many QA rows the *whole* evaluation decodes: the budget is split
        into contiguous windows of the global row order (see :func:`row_window`), so the
        scored rows and the noise level stay the same when the rank count changes. Decoding
        is orders of magnitude more expensive than one teacher-forced forward, and a rank
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
        samples = 0
        first_token_hit = 0
        enabled = is_main_process()
        world, rank = _distributed_world_size(), _distributed_rank()
        if max_qa is None:
            window, row_source = (0, None), loader
        elif row_loader is None and world > 1:
            print(
                f"[evaluator] WARNING: evaluation.autoregressive_max_qa={max_qa} is a global "
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
        desc = "Validation (autoregressive)"
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
            rows = batch["question_ids"].size(0)
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
            generated = model.generate_answers_with_prefix(
                prefix,
                q,
                ids["question_ids"][rows_slice],
                ids["question_mask"][rows_slice],
                self.tokenizer,
                self.cfg.evaluation.max_new_tokens,
                group_by_context=True,
                max_rows_per_group=max(1, self.cfg.evaluation.qa_batch_size),
            )
            for i, record in enumerate(records):
                row = generated[i].tolist()
                first_token_hit += int(
                    bool(row) and self._first_token_hit(row[0], record)
                )
                metrics = best_reference_metrics(
                    answer_line(self.tokenizer.decode(row, skip_special_tokens=True)),
                    record.references,
                )
                for key, value in metrics.items():
                    sums[key] += value
                samples += 1
            progress.set_postfix({"qa_rows": samples}, refresh=False)
            if window[1] is not None and cursor >= window[1]:
                break
        progress.close()
        model.train(was_training)
        first_token_hit, samples = _distributed_sum([first_token_hit, samples], device)
        sums = dict(
            zip(METRIC_KEYS, _distributed_sum([sums[key] for key in METRIC_KEYS], device))
        )
        result = {key: value / max(1, samples) for key, value in sums.items()}
        # Same retrieval diagnostic as in teacher_forced, but measured on tokens the
        # model actually produced.  This is the number to compare against
        # teacher-forced first_token_em; a large gap between them means the model
        # never learned to emit the answer from memory.
        result["first_token_em"] = first_token_hit / max(1, samples)
        if include_teacher_metrics:
            # Costs a second full pass over the *sharded* loader: the teacher-forced numbers
            # come from its own all-reduce, so they must not be computed on the unsharded row
            # loader (every rank would score every row and the reduce would count each row
            # world_size times).  Callers that already run teacher_forced separately
            # (scripts/test.py) should pass False.
            teacher = self.teacher_forced(model, loader, device)
            result.update(
                {
                    "ppl": teacher["ppl"],
                    "qa_loss": teacher["qa_loss"],
                    **{key: teacher[key] for key in ("embedding_recon_loss", "token_recon_loss", "causal_recon_loss", "distill_loss")},
                }
            )
        return result
