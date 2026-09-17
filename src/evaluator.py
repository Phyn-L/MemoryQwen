from __future__ import annotations

import math

import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from utils.ddp import is_main_process

from .losses import combine_losses, context_lm_loss, qa_loss, reconstruction_loss
from .metrics import METRIC_KEYS, best_reference_metrics, normalize_answer


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
        """The objective that shapes the memory, whichever one the config selected.

        Both variants are reported under ``reconstruction_loss`` so that run
        dashboards stay comparable across the two objectives. The context-LM value is
        nats per context token and therefore not numerically comparable with the
        embedding-regression value; only its trend matters.
        """
        if self.cfg.memory.reconstruction_loss == "context_lm":
            terms = model.context_lm_terms(
                ids["context_ids"], ids["context_mask"], prefix.layer_memory,
                self.cfg.memory.context_lm_positions,
            )
            return context_lm_loss(
                terms.hidden, terms.labels, model.context_lm_head, terms.mask,
            )
        return reconstruction_loss(
            prefix.reconstruction, prefix.context_target, prefix.context_mask,
            self.cfg.memory.reconstruction_cosine_weight,
            self.cfg.memory.reconstruction_loss,
        )

    @torch.no_grad()
    def teacher_forced(self, model, loader, device):
        model.eval()
        qa_sum = reconstruction_sum = 0.0
        totals = {key: 0.0 for key in METRIC_KEYS}
        samples = 0
        first_token_hits = 0
        qa_weight = context_weight = 0
        enabled = is_main_process()
        for batch in tqdm(
            loader,
            total=len(loader),
            desc="Validation (teacher-forced)",
            unit="batch",
            disable=not enabled,
            leave=False,
        ):
            ids = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "records"
            }
            embed = model.qwen.get_input_embeddings()
            prefix = model.encode_context_prefix(
                embed(ids["context_ids"]), ids["context_mask"]
            )
            reconstruction = self.auxiliary_loss(model, prefix, ids)
            contexts = ids["context_ids"].size(0)
            reconstruction_sum += float(reconstruction) * contexts
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
                    metrics = best_reference_metrics(text, record.references)
                    for key, value in metrics.items():
                        totals[key] += value
                    samples += 1
        model.train()
        qa_sum, reconstruction_sum, first_token_hits, qa_weight, context_weight, samples = (
            _distributed_sum(
                [qa_sum, reconstruction_sum, first_token_hits, qa_weight, context_weight, samples],
                device,
            )
        )
        metric_totals = dict(
            zip(METRIC_KEYS, _distributed_sum([totals[key] for key in METRIC_KEYS], device))
        )
        qa_value = qa_sum / max(1, qa_weight)
        reconstruction_value = reconstruction_sum / max(1, context_weight)
        total_value = (
            self.cfg.memory.qa_weight * qa_value
            + self.cfg.memory.reconstruction_weight * reconstruction_value
        )
        return {
            "qa_loss": qa_value,
            "ppl": math.exp(min(qa_value, 20)),
            "reconstruction_loss": reconstruction_value,
            "loss": total_value,
            **{key: value / max(1, samples) for key, value in metric_totals.items()},
            # Not an answer-quality score: teacher forcing feeds the gold answer
            # prefix, so every metric here mostly measures lexical continuation.
            # Treat first_token_em as the retrieval signal and prefer the
            # autoregressive numbers as the headline result.
            "first_token_em": first_token_hits / max(1, samples),
        }

    @torch.no_grad()
    def autoregressive(self, model, loader, device, include_teacher_metrics=True, max_qa=None):
        """Headline evaluation: every answer token is produced by the model itself.

        Unlike :meth:`teacher_forced` this never feeds the gold answer back, so these are
        the numbers to quote. The metrics use the single official SQuAD normalizer from
        ``src.metrics``, which is also what ``scripts/test_icl_baseline.py`` scores with, so
        the two numbers are directly comparable. ``include_teacher_metrics`` additionally
        reports the teacher-forced ppl/qa_loss/reconstruction_loss at the cost of a second
        pass.

        ``max_qa`` caps how many QA rows *this rank* decodes.  Decoding is orders of
        magnitude more expensive than one teacher-forced forward, and a rank that stays
        inside the evaluation for longer than the NCCL watchdog timeout (10 minutes by
        default) makes every other rank abort with "Watchdog caught collective
        operation timeout". The cap is per rank, so the global number of decoded rows is
        ``max_qa * world_size``.
        """
        model.eval()
        sums = {key: 0.0 for key in METRIC_KEYS}
        samples = 0
        first_token_hit = 0
        enabled = is_main_process()
        for batch in tqdm(
            loader,
            total=len(loader),
            desc="Validation (autoregressive)",
            unit="batch",
            disable=not enabled,
            leave=False,
        ):
            if max_qa is not None and samples >= max_qa:
                break
            ids = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "records"
            }
            q = ids["qa_context_indices"]
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
                ids["question_ids"],
                ids["question_mask"],
                self.tokenizer,
                self.cfg.evaluation.max_new_tokens,
                group_by_context=True,
                max_rows_per_group=max(1, self.cfg.evaluation.qa_batch_size),
            )
            for i, record in enumerate(batch["records"]):
                if max_qa is not None and samples >= max_qa:
                    break
                row = generated[i].tolist()
                first_token_hit += int(
                    bool(row) and self._first_token_hit(row[0], record)
                )
                metrics = best_reference_metrics(
                    self.tokenizer.decode(row, skip_special_tokens=True),
                    record.references,
                )
                for key, value in metrics.items():
                    sums[key] += value
                samples += 1
        model.train()
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
            # Costs a second full pass over the loader.  Callers that already run
            # teacher_forced separately (scripts/test.py) should pass False.
            teacher = self.teacher_forced(model, loader, device)
            result.update(
                {
                    "ppl": teacher["ppl"],
                    "qa_loss": teacher["qa_loss"],
                    "reconstruction_loss": teacher["reconstruction_loss"],
                }
            )
        return result
