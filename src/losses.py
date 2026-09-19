import torch
import torch.nn.functional as F

from .dtypes import no_autocast

def qa_loss(logits, labels):
    # Reduce in float32: the backbone may emit bfloat16 logits, and the answer-only
    # cross entropy is small enough that bfloat16 rounding is visible in the loss.
    token_logits = logits[..., :-1, :].float()
    token_labels = labels[..., 1:]
    flat_loss = F.cross_entropy(token_logits.reshape(-1, logits.size(-1)), token_labels.reshape(-1), ignore_index=-100, reduction="none")
    active = token_labels.reshape(-1).ne(-100)
    if not active.any():
        return token_logits.sum() * 0.0
    token_loss = flat_loss.view(token_labels.shape)
    row_counts = active.view(token_labels.shape).sum(dim=1).clamp_min(1)
    row_loss = token_loss.sum(dim=1) / row_counts
    return row_loss.mean()
def recon_loss(predicted,target,mask=None,cosine_weight=0.1,mode="mse_cosine"):
    if predicted.ndim == target.ndim + 1:
        target=target.unsqueeze(1).expand_as(predicted)
    if predicted.shape != target.shape: raise ValueError(f"recon shape mismatch: {predicted.shape} vs {target.shape}")
    # The decoders may run in float32 while the context target is the backbone's
    # bfloat16 input embedding; reduce in float32 so the comparison is meaningful.
    predicted, target = predicted.float(), target.float()
    e=(predicted-target).pow(2).mean(-1)
    if mask is None: mask=torch.ones_like(e,dtype=torch.bool)
    if mask.ndim == 2: mask=mask.unsqueeze(1).expand_as(e)
    mask=mask.to(e.device,dtype=e.dtype)
    # Reduce token/layer loss per context first, then average contexts.
    mse_per_context = (e * mask).sum(dim=tuple(range(1, e.ndim))) / mask.sum(dim=tuple(range(1, e.ndim))).clamp_min(1)
    mse = mse_per_context.mean()
    cosine=1-F.cosine_similarity(predicted,target,dim=-1)
    cosine_per_context = (cosine * mask).sum(dim=tuple(range(1, cosine.ndim))) / mask.sum(dim=tuple(range(1, cosine.ndim))).clamp_min(1)
    cosine = cosine_per_context.mean()
    if mode == "mse": return mse
    if mode == "cosine": return cosine
    return mse + cosine_weight*cosine

def _chunk_cross_entropy(hidden_chunk, labels_chunk, weight):
    """Vocabulary cross entropy for one flat chunk. Separated so it can be checkpointed.

    ``weight`` is the already materialised ``[vocab, D]`` unembedding, not the head
    module: a tied head materialises its weight lazily, and letting that happen inside
    the checkpointed function makes the recomputation save a different number of
    tensors than the original forward (``CheckpointError``). Passing the weight in as an
    input keeps the recomputation pure and the materialisation in the outer graph.
    """
    logits = F.linear(hidden_chunk, weight).float()
    return F.cross_entropy(logits, labels_chunk, ignore_index=-100, reduction="sum")


def token_recon_loss(hidden, labels, head, mask=None, max_logits_rows=256):
    """Context next-token prediction through the memory.

    ``hidden`` is ``[B, num_layers, P, D]``: for every Qwen layer, the bottleneck
    hidden state of the memory decoder at ``P`` sampled context positions.
    ``labels`` is ``[B, P]`` with the context token id that each position must
    predict (``-100`` where the position was padding and must be ignored).
    ``head`` is the shared ``Linear(D, vocab_size)`` unembedding.

    The loss is the mean over layers of the token cross entropy, i.e. nats per
    context token. Unlike the embedding regression it has no "predict the mean
    vector" optimum: the memory has to discriminate the actual token from the rest
    of the vocabulary.

    Logits are produced one layer (and at most ``max_logits_rows`` positions) at a
    time. Materialising ``[B, num_layers, P, vocab_size]`` at once would need
    gigabytes, and the head is cheap enough that chunking is nearly free.

    Each chunk is wrapped in ``torch.utils.checkpoint`` when gradients are enabled.
    Without it autograd keeps every chunk's logits alive for the backward pass --
    ``num_layers * max_logits_rows * vocab_size`` floats, which is ~4.4 GB for the
    Qwen3-1.7B config (28 x 256 x 151936 x 4 bytes) and dominated the measured peak.
    Recomputing the head in the backward pass costs one extra head forward (~8% of the
    backbone) and removes essentially all of that.
    """
    if hidden.ndim != 4:
        raise ValueError("token_recon hidden must have shape [B, num_layers, P, D]")
    if labels.shape != hidden.shape[:1] + hidden.shape[2:3]:
        raise ValueError(f"token_recon labels {tuple(labels.shape)} do not match hidden {tuple(hidden.shape)}")
    if mask is None:
        mask = labels.ne(-100)
    else:
        mask = mask.bool() & labels.ne(-100)
    labels = labels.masked_fill(~mask, -100)
    # A plain nn.Linear exposes ``weight``; src.model.VocabularyHead in tied mode does
    # not and reports the dtype its adapter computes in instead.
    head_dtype = getattr(head, "compute_dtype", None) or head.weight.dtype
    # Materialise the unembedding once per call, outside the checkpointed chunks: a tied
    # head builds ``E @ adapter`` lazily, and doing that inside the checkpoint would make
    # the recomputation save a different number of tensors than the forward.
    weight = (
        head.materialized_weight()
        if hasattr(head, "materialized_weight") else head.weight
    )
    total = None
    layers = hidden.size(1)
    for layer in range(layers):
        layer_hidden = hidden[:, layer]                       # [B, P, D]
        flat_hidden = layer_hidden.reshape(-1, layer_hidden.size(-1)).to(head_dtype)
        flat_labels = labels.reshape(-1)
        rows = flat_hidden.size(0)
        step = max(1, min(int(max_logits_rows), rows))
        with no_autocast(hidden.device):
            layer_sum = None
            layer_count = 0
            for start in range(0, rows, step):
                chunk_hidden = flat_hidden[start:start + step]
                chunk_labels = flat_labels[start:start + step]
                count = chunk_labels.ne(-100).sum()
                if int(count) == 0:
                    continue
                if torch.is_grad_enabled() and chunk_hidden.requires_grad:
                    chunk_sum = torch.utils.checkpoint.checkpoint(
                        _chunk_cross_entropy, chunk_hidden, chunk_labels, weight,
                        use_reentrant=False,
                    )
                else:
                    chunk_sum = _chunk_cross_entropy(chunk_hidden, chunk_labels, weight)
                layer_sum = chunk_sum if layer_sum is None else layer_sum + chunk_sum
                layer_count += int(count)
            if layer_sum is None:
                continue
            layer_loss = layer_sum / layer_count
        total = layer_loss if total is None else total + layer_loss
    if total is None:
        return hidden.sum() * 0.0
    return total / layers


def sample_positions(mask, count, generator=None):
    """Sample up to ``count`` True positions of ``mask`` ``[B, S]`` per row.

    Same contract as ``MetaLoRA.sample_context_targets`` but standalone, so a caller can
    index an already-aligned ``(hidden, labels)`` pair (see :func:`sequence_lm_loss`).
    Returns ``[B, P]`` indices and a ``[B, P]`` keep-mask; rows with fewer than ``count``
    valid positions get ``-1`` in the unused slots.

    ``-1`` rather than "repeat index 0": a repeated 0 is indistinguishable from a row that
    legitimately sampled position 0, so a caller that forgot the keep-mask would score the
    same token ``count - valid`` extra times (a context shorter than the budget would be
    trained mostly on its own first token). The losses translate a negative position into
    ``ignore_index``, which makes that mistake impossible.
    """
    mask = mask.bool()
    if mask.ndim != 2:
        raise ValueError("sample_positions expects a [B, S] mask")
    device = mask.device
    scores = torch.rand(mask.shape, device=device, generator=generator).masked_fill(~mask, float("-inf"))
    budget = max(1, min(int(count), mask.size(1)))
    positions = scores.topk(budget, dim=1).indices
    keep = mask.gather(1, positions)
    return positions.masked_fill(~keep, -1), keep


def sequence_lm_loss(hidden, labels, head, mask=None, positions=None, max_logits_rows=256):
    """Next-token cross entropy over a whole sequence (the autoencoding objective).

    ``hidden`` is ``[B, S, H]``: the backbone's own last hidden state for a sequence whose
    keys/values were prefixed by the memory. ``hidden[i]`` scores ``labels[i]`` **directly**
    -- this function does *not* shift, unlike :func:`qa_loss` -- so the caller shifts once
    and any ``positions`` it passes index the same aligned pair that is scored.

    Unlike :func:`token_recon_loss` the vocabulary head is applied once (the sequence has one
    hidden state per position, not one per layer), so scoring every position costs a single
    head pass. ``positions`` gathers first, then chunks and checkpoints, so a dense
    objective still respects the logits budget.

    The head is materialised outside the checkpointed chunks for the same reason as in
    :func:`token_recon_loss`: a lazily materialised weight would make the recomputation save
    a different number of tensors than the forward.
    """
    if hidden.ndim != 3:
        raise ValueError("sequence_lm hidden must have shape [B, S, H]")
    if labels.shape != hidden.shape[:2]:
        raise ValueError(f"sequence_lm labels {tuple(labels.shape)} do not match hidden {tuple(hidden.shape)}")
    if mask is None:
        mask = labels.ne(-100)
    else:
        mask = mask.bool() & labels.ne(-100)
    if positions is not None:
        # `sample_positions` writes -1 into the slots it could not fill (a row with fewer
        # valid positions than the budget). Those must not be scored: gathering them with a
        # clamped index and ANDing in "was this slot real" is what drops them. Relying on
        # the gathered mask alone is not enough, because a duplicate would point at a
        # position that is itself valid.
        valid = positions >= 0
        safe = positions.clamp_min(0)
        index = safe.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        hidden = hidden.gather(1, index)
        labels = labels.gather(1, safe)
        mask = mask.gather(1, safe) & valid
    labels = labels.masked_fill(~mask, -100)
    head_dtype = getattr(head, "compute_dtype", None) or head.weight.dtype
    weight = (
        head.materialized_weight()
        if hasattr(head, "materialized_weight") else head.weight
    )
    flat_hidden = hidden.reshape(-1, hidden.size(-1)).to(head_dtype)
    flat_labels = labels.reshape(-1)
    rows = flat_hidden.size(0)
    step = max(1, min(int(max_logits_rows), rows)) if rows else 1
    total = None
    count = 0
    with no_autocast(hidden.device):
        for start in range(0, rows, step):
            chunk_hidden = flat_hidden[start:start + step]
            chunk_labels = flat_labels[start:start + step]
            active = int(chunk_labels.ne(-100).sum())
            if active == 0:
                continue
            if torch.is_grad_enabled() and chunk_hidden.requires_grad:
                chunk_sum = torch.utils.checkpoint.checkpoint(
                    _chunk_cross_entropy, chunk_hidden, chunk_labels, weight,
                    use_reentrant=False,
                )
            else:
                chunk_sum = _chunk_cross_entropy(chunk_hidden, chunk_labels, weight)
            total = chunk_sum if total is None else total + chunk_sum
            count += active
    if total is None or count == 0:
        return hidden.sum() * 0.0
    return total / count


def kl_distill_loss(student_hidden, teacher_hidden, head, mask=None, temperature=1.0,
                    positions=None, max_logits_rows=256, topk=0, entropy_weight=False):
    """Distil the full-context distribution into the memory-conditioned one.

    ``teacher_hidden`` is the plain causal-LM state at the same token positions (the encoder
    pass's context rows, i.e. the full-context bypass), ``student_hidden`` is the
    memory-conditioned state (:meth:`MetaLoRA.autoencode_with_memory`), and both are scored
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
    head_dtype = getattr(head, "compute_dtype", None) or head.weight.dtype
    weight = (
        head.materialized_weight()
        if hasattr(head, "materialized_weight") else head.weight
    )
    if positions is not None:
        # See sequence_lm_loss: -1 means "this sampled slot could not be filled" and must
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


def combine_losses(qa, recon, qa_weight=1., recon_weight=1., ae=None, ae_weight=0., distill=None, distill_weight=0.):
    total=qa_weight*qa+recon_weight*recon
    terms={"loss":total,"qa_loss":qa,"recon_loss":recon}
    if ae is not None and ae_weight:
        total=total+ae_weight*ae; terms["causal_recon_loss"]=ae
    if distill is not None and distill_weight:
        total=total+distill_weight*distill; terms["distill_loss"]=distill
    terms["loss"]=total
    return total,terms


def memory_objectives(model, prefix, context_ids, context_mask, cfg):
    """Unweighted context-level losses; disabled branches do no decoder work."""
    zero = prefix.memory.sum() * 0.0
    terms = {name: zero for name in ("embedding_recon_loss", "token_recon_loss", "causal_recon_loss", "distill_loss")}
    if cfg.embedding_recon_weight:
        terms["embedding_recon_loss"] = recon_loss(prefix.recon, prefix.context_target, context_mask, cfg.embedding_recon_cosine_weight, cfg.embedding_recon_loss)
    if cfg.token_recon_weight:
        t = model.token_recon_terms(context_ids, context_mask, prefix.layer_memory, cfg.token_recon_positions)
        terms["token_recon_loss"] = token_recon_loss(t.hidden, t.labels, model.token_recon_head, t.mask)
    if cfg.causal_recon_weight or cfg.distill_weight:
        hidden = model.autoencode_with_memory(prefix, model.qwen.get_input_embeddings()(context_ids), context_mask)[:, :-1]
        mask = context_mask[:, 1:]
        if cfg.causal_recon_weight:
            positions = sample_positions(mask, cfg.causal_recon_positions)[0] if cfg.causal_recon_positions else None
            terms["causal_recon_loss"] = sequence_lm_loss(hidden, context_ids[:, 1:], model.ae_head, mask, positions=positions)
        if cfg.distill_weight:
            positions = sample_positions(mask, cfg.distill_positions)[0] if cfg.distill_positions else None
            terms["distill_loss"] = kl_distill_loss(hidden, prefix.context_hidden[:, :-1], model.ae_head, mask, cfg.distill_temperature, positions=positions, topk=cfg.distill_topk, entropy_weight=cfg.distill_entropy_weight)
    return terms


def objective_total(qa, terms, cfg):
    return cfg.qa_weight * qa + sum(getattr(cfg, name.removesuffix("_loss") + "_weight") * loss for name, loss in terms.items())
