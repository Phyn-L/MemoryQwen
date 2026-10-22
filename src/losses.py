"""Objectives: per-task cross entropy and teacher-forced reconstruction statistics.

The whole batch is projected through the vocabulary head in one *flattened* pass (chunked by
``loss.max_logits_rows``) and each row is normalised by its own active-token count, so
"average tokens within a row, then average rows within a task" holds without one Python
iteration per row.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .dtypes import no_autocast


def _resolve_mask(labels, mask):
    if mask is None:
        return labels.ne(-100)
    return mask.bool() & labels.ne(-100)


def _head_weight(head):
    """``[vocab, D]`` unembedding, materialised outside checkpointed chunks."""
    weight = (
        head.materialized_weight() if hasattr(head, "materialized_weight") else head.weight
    )
    return weight, (getattr(head, "compute_dtype", None) or weight.dtype)


def _chunk_row_cross_entropy(hidden_chunk, labels_chunk, weight, rows_chunk, row_count):
    """Per-row summed cross entropy for one flat chunk. Separated so it can be checkpointed.

    ``weight`` is the already materialised ``[vocab, D]`` unembedding, not the head module: a
    tied head materialises its weight lazily, and letting that happen inside the checkpointed
    function makes the recomputation save a different number of tensors than the original
    forward (``CheckpointError``). Passing the weight in as an input keeps the recomputation
    pure and the materialisation in the outer graph.
    """
    logits = F.linear(hidden_chunk, weight).float()
    token_loss = F.cross_entropy(
        logits, labels_chunk, ignore_index=-100, reduction="none"
    )
    active = labels_chunk.ne(-100)
    partial = token_loss.new_zeros(row_count)
    partial.index_add_(0, rows_chunk, token_loss.masked_fill(~active, 0.0))
    return partial


def _row_statistics(hidden, labels, head, max_logits_rows, with_correct):
    """Chunked vocabulary projection; returns per-row ``(nll_sum, correct, tokens)``.

    Every row of the batch is projected in one pass: the ``[R, S]`` pair is flattened to
    ``R * S`` rows and cut into ``max_logits_rows`` chunks that may straddle row boundaries,
    because the per-row reduction is an ``index_add_`` over the row index of each token.
    """
    if hidden.ndim != 3:
        raise ValueError("a supervision row must have shape [B, S, H]")
    if labels.shape != hidden.shape[:2]:
        raise ValueError(
            f"labels {tuple(labels.shape)} do not match hidden {tuple(hidden.shape)}"
        )
    row_count, width = labels.shape
    weight, head_dtype = _head_weight(head)
    # Labels stay integer (cross_entropy requires it); the hidden states are what has to match
    # the head's dtype.
    flat_labels = labels.reshape(-1)
    flat_hidden = hidden.reshape(-1, hidden.size(-1)).to(head_dtype)
    active = flat_labels.ne(-100)
    tokens = active.view(row_count, width).sum(dim=1).float()
    rows_of_token = torch.arange(row_count, device=labels.device).repeat_interleave(width)
    total_tokens = flat_hidden.size(0)
    step = max(1, min(int(max_logits_rows), total_tokens)) if total_tokens else 1

    nll = torch.zeros(row_count, dtype=torch.float32, device=labels.device)
    correct = (
        torch.zeros(row_count, dtype=torch.float32, device=labels.device)
        if with_correct
        else None
    )
    if total_tokens:
        with no_autocast(hidden.device):
            for start in range(0, total_tokens, step):
                stop = start + step
                chunk_hidden = flat_hidden[start:stop]
                chunk_labels = flat_labels[start:stop]
                chunk_rows = rows_of_token[start:stop]
                if not bool(chunk_labels.ne(-100).any()):
                    continue
                if with_correct and not (
                    torch.is_grad_enabled() and chunk_hidden.requires_grad
                ):
                    logits = F.linear(chunk_hidden, weight).float()
                    token_loss = F.cross_entropy(
                        logits, chunk_labels, ignore_index=-100, reduction="none"
                    )
                    partial = token_loss.new_zeros(row_count)
                    partial.index_add_(
                        0, chunk_rows,
                        token_loss.masked_fill(chunk_labels.eq(-100), 0.0),
                    )
                    prediction = logits.argmax(dim=-1)
                    hit = prediction.eq(chunk_labels) & chunk_labels.ne(-100)
                    correct.index_add_(0, chunk_rows, hit.float())
                elif torch.is_grad_enabled() and chunk_hidden.requires_grad:
                    partial = torch.utils.checkpoint.checkpoint(
                        _chunk_row_cross_entropy,
                        chunk_hidden,
                        chunk_labels,
                        weight,
                        chunk_rows,
                        row_count,
                        use_reentrant=False,
                    )
                else:
                    partial = _chunk_row_cross_entropy(
                        chunk_hidden, chunk_labels, weight, chunk_rows, row_count
                    )
                nll = nll + partial.float()
                if with_correct and torch.is_grad_enabled() and chunk_hidden.requires_grad:
                    prediction = F.linear(chunk_hidden, weight).argmax(dim=-1)
                    hit = prediction.eq(chunk_labels) & chunk_labels.ne(-100)
                    correct.index_add_(0, chunk_rows, hit.float())
    return nll, correct, tokens


def sequence_lm_loss(
    hidden, labels, head, mask=None, max_logits_rows=256,
):
    """Per-row mean next-token cross entropy over a whole sequence.

    ``hidden`` is ``[B, S, H]``: the backbone's own last hidden state for a sequence whose
    keys/values were prefixed by the memory. ``hidden[i]`` scores ``labels[i]`` **directly**
    -- this function does *not* shift, so the caller shifts once.

    Returns ``(row_losses, row_mask)``: the per-row mean over that row's active tokens (0 for
    a row with none) and a boolean ``[B]`` mask telling the caller which rows contributed.
    ``supervision_objectives`` uses the mask so an empty row cannot dilute a task's mean.
    """
    if labels.shape[:2] != hidden.shape[:2]:
        raise ValueError(
            f"labels {tuple(labels.shape[:2])} do not match hidden {tuple(hidden.shape[:2])}"
        )
    mask = _resolve_mask(labels, mask)
    labels = labels.masked_fill(~mask, -100)
    nll, _, tokens = _row_statistics(
        hidden, labels, head, max_logits_rows, with_correct=False
    )
    row_mask = tokens > 0
    row_losses = nll / tokens.clamp_min(1.0)
    return row_losses, row_mask


def teacher_forced_row_stats(hidden, labels, head, max_logits_rows=256):
    """Per-row token count, exact-match count and summed NLL of one aligned batch.

    Used by the evaluation-only reconstruction metric: the same memory-prefixed hidden states
    the training loss scores, reduced to the three numbers an accuracy / perplexity needs. No
    generation, so it costs one forward instead of ``max_context_tokens`` decodes.
    """
    mask = _resolve_mask(labels, None)
    labels = labels.masked_fill(~mask, -100)
    nll, correct, tokens = _row_statistics(
        hidden, labels, head, max_logits_rows, with_correct=True
    )
    return {"nll": nll, "correct": correct, "tokens": tokens}


def kl_distill_loss(student_hidden, teacher_hidden, head, mask=None, temperature=1.0,
                    positions=None, max_logits_rows=256, topk=0, entropy_weight=False):
    """Distil the full-context distribution into the memory-conditioned one.

    ``teacher_hidden`` is the plain causal-LM state at the same token positions (the encoder
    pass's context rows, i.e. the full-context bypass), ``student_hidden`` is the
    memory-conditioned state (:meth:`ContextMemoryModel._distillation_student_hidden`), and both are scored
    by the same unembedding. The loss is ``T^2 * KL(p_teacher || p_student)`` -- forward KL,
    so the student is pushed to cover the teacher's mass -- averaged over the scored
    positions. The teacher never receives gradient.

    Why not cross entropy on the hard label: most context tokens are predictable from the
    language-model prior alone, so a one-hot target spends memory capacity on content the
    frozen model already knows (ICAE measures normal text at BLEU 99.3 versus 3.5 for
    patterned-random and 0.2 for random text). A distribution target preserves the *residual*
    uncertainty, which is exactly what the memory has to carry. ``entropy_weight`` weights
    each position by the teacher's own entropy; ``topk`` truncates the teacher to its k most
    likely tokens to bound the softmax cost.

    Memory note: the scored positions' logits stay alive for the backward pass (one fp32
    ``[rows, vocab]`` chunk per active chunk), so keep ``max_logits_rows`` and the position
    budget modest -- 256 positions at 256 rows/step is ~155 MB per row of the batch.
    """
    if student_hidden.ndim != 3:
        raise ValueError("kl_distill student hidden must have shape [B, S, H]")
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError(
            f"kl_distill needs matching hidden states, got {tuple(student_hidden.shape)} "
            f"and {tuple(teacher_hidden.shape)}"
        )
    if temperature <= 0:
        raise ValueError("kl_distill temperature must be positive")
    if topk < 0:
        raise ValueError("kl_distill topk must be >= 0")
    if mask is None:
        mask = torch.ones(student_hidden.shape[:2], dtype=torch.bool, device=student_hidden.device)
    else:
        mask = mask.bool()
    weight, head_dtype = _head_weight(head)
    if positions is not None:
        # See sample_positions: -1 means "this sampled slot could not be filled" and must
        # not be scored, which the gathered mask alone cannot express.
        valid = positions >= 0
        safe = positions.clamp_min(0)
        index = safe.unsqueeze(-1).expand(-1, -1, student_hidden.size(-1))
        student_hidden = student_hidden.gather(1, index)
        teacher_hidden = teacher_hidden.gather(1, index)
        mask = mask.gather(1, safe) & valid
    width = student_hidden.size(-1)
    flat_student = student_hidden.reshape(-1, width).to(head_dtype)
    flat_teacher = teacher_hidden.reshape(-1, width).to(head_dtype)
    flat_mask = mask.reshape(-1)
    rows = flat_student.size(0)
    step = max(1, min(int(max_logits_rows), rows)) if rows else 1
    scale = float(temperature) ** 2
    total = None
    weight_sum = 0.0
    with no_autocast(student_hidden.device):
        for start in range(0, rows, step):
            active = flat_mask[start:start + step]
            if not bool(active.any()):
                continue
            student_logits = F.linear(flat_student[start:start + step], weight).float()
            with torch.no_grad():
                teacher_logits = F.linear(flat_teacher[start:start + step], weight).float()
                if topk and topk < teacher_logits.size(-1):
                    threshold = teacher_logits.topk(topk, dim=-1).values[:, -1:]
                    teacher_logits = teacher_logits.masked_fill(teacher_logits < threshold, float("-inf"))
                teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)
                if entropy_weight:
                    entropy = -(teacher_prob * torch.log(teacher_prob.clamp_min(1e-9))).sum(-1)
                    per_row_weight = entropy * active.to(entropy.dtype)
                else:
                    per_row_weight = active.to(teacher_prob.dtype)
            student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
            per_row = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(-1) * scale
            contribution = (per_row * per_row_weight).sum()
            total = contribution if total is None else total + contribution
            weight_sum += float(per_row_weight.sum())
    if total is None or weight_sum == 0.0:
        return student_hidden.sum() * 0.0
    return total / weight_sum


def memory_objectives(model, prefix, context_ids, context_mask, cfg, generator=None):
    """Unweighted context-level losses; disabled branches do no decoder work.

    ``generator`` drives the distillation position sampling (see :func:`sample_positions`).
    """
    zero = prefix.memory.sum() * 0.0
    terms = {"distill_loss": zero}
    if cfg.distill_weight:
        hidden = model._distillation_student_hidden(
            prefix, model.qwen.get_input_embeddings()(context_ids), context_mask
        )[:, :-1]
        mask = context_mask[:, 1:]
        positions = (
            sample_positions(mask, cfg.distill_positions, generator=generator)[0]
            if cfg.distill_positions
            else None
        )
        terms["distill_loss"] = kl_distill_loss(
            hidden,
            prefix.context_hidden[:, :-1],
            model.distill_head,
            mask,
            cfg.distill_temperature,
            positions=positions,
            max_logits_rows=cfg.max_logits_rows,
            topk=cfg.distill_topk,
            entropy_weight=cfg.distill_entropy_weight,
        )
    return terms


def objective_total(qa, terms, cfg):
    return cfg.qa_weight * qa + sum(
        getattr(cfg, name.removesuffix("_loss") + "_weight") * loss
        for name, loss in terms.items()
    )


def scheduled_task_weights(step, total_steps, cfg, stage=None):
    """Return the QA and Recon sampling probabilities at one optimizer step."""
    if cfg.train_stage == "only_qa" or stage == "qa":
        return 1.0, 0.0
    if cfg.train_stage == "only_recon" or stage == "recon":
        return 0.0, 1.0
    progress = min(max(step, 0) / max(total_steps - 1, 1), 1.0)
    if cfg.recon_schedule_curve == "cosine":
        progress = (1.0 - math.cos(progress * math.pi)) / 2.0

    def interpolate(schedule):
        start, end = schedule
        return start + (end - start) * progress

    return interpolate(cfg.qa_schedule), interpolate(cfg.recon_schedule)


def supervision_objectives(
    row_losses, task_ids, cfg, qa_weight=None, recon_weight=None, row_mask=None,
):
    """Per-task means with externally scheduled task weights.

    ``row_mask`` selects the supervision rows that actually carried a target; without it a
    row with no active token (which scores 0) would drag its task's mean towards zero.
    """
    recon_mask, qa_mask = task_ids.eq(0), task_ids.eq(1)
    if row_mask is not None:
        row_mask = row_mask.to(task_ids.device, dtype=torch.bool)
        if row_mask.shape != task_ids.shape:
            raise ValueError(
                f"row_mask {tuple(row_mask.shape)} does not match task ids {tuple(task_ids.shape)}"
            )
        qa_mask, recon_mask = qa_mask & row_mask, recon_mask & row_mask
    qa_weight = cfg.qa_weight if qa_weight is None else qa_weight
    recon_weight = cfg.recon_weight if recon_weight is None else recon_weight
    zero = row_losses.sum() * 0
    qa = row_losses[qa_mask].mean() if bool(qa_mask.any()) else zero
    recon = row_losses[recon_mask].mean() if bool(recon_mask.any()) else zero
    return qa_weight * qa + recon_weight * recon, {"qa_loss": qa, "recon_loss": recon}
