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
def reconstruction_loss(predicted,target,mask=None,cosine_weight=0.1,mode="mse_cosine"):
    if predicted.ndim == target.ndim + 1:
        target=target.unsqueeze(1).expand_as(predicted)
    if predicted.shape != target.shape: raise ValueError(f"reconstruction shape mismatch: {predicted.shape} vs {target.shape}")
    # The decoders may run in float32 while the context target is the backbone's
    # bfloat16 input embedding; reduce in float32 so the comparison is meaningful.
    predicted, target = predicted.float(), target.float()
    e=(predicted-target).pow(2).mean(-1)
    if mask is None: mask=torch.ones_like(e,dtype=torch.bool)
    if mask.ndim == 2: mask=mask.unsqueeze(1).expand_as(e)
    mask=mask.to(e.device,dtype=e.dtype)
    # Reduce token/layer loss per context first, then average contexts.
    context_mask = mask.any(dim=1) if mask.ndim == 3 else mask
    mse_per_context = (e * mask).sum(dim=tuple(range(1, e.ndim))) / mask.sum(dim=tuple(range(1, e.ndim))).clamp_min(1)
    mse = mse_per_context.mean()
    cosine=1-F.cosine_similarity(predicted,target,dim=-1)
    cosine_per_context = (cosine * mask).sum(dim=tuple(range(1, cosine.ndim))) / mask.sum(dim=tuple(range(1, cosine.ndim))).clamp_min(1)
    cosine = cosine_per_context.mean()
    if mode == "mse": return mse
    if mode == "cosine": return cosine
    return mse + cosine_weight*cosine

def _chunk_cross_entropy(hidden_chunk, labels_chunk, head):
    """Vocabulary cross entropy for one flat chunk. Separated so it can be checkpointed."""
    logits = head(hidden_chunk).float()
    return F.cross_entropy(logits, labels_chunk, ignore_index=-100, reduction="sum")


def context_lm_loss(hidden, labels, head, mask=None, max_logits_rows=256):
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
        raise ValueError("context_lm hidden must have shape [B, num_layers, P, D]")
    if labels.shape != hidden.shape[:1] + hidden.shape[2:3]:
        raise ValueError(f"context_lm labels {tuple(labels.shape)} do not match hidden {tuple(hidden.shape)}")
    if mask is None:
        mask = labels.ne(-100)
    else:
        mask = mask.bool() & labels.ne(-100)
    labels = labels.masked_fill(~mask, -100)
    head_dtype = head.weight.dtype
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
                        _chunk_cross_entropy, chunk_hidden, chunk_labels, head,
                        use_reentrant=False,
                    )
                else:
                    chunk_sum = _chunk_cross_entropy(chunk_hidden, chunk_labels, head)
                layer_sum = chunk_sum if layer_sum is None else layer_sum + chunk_sum
                layer_count += int(count)
            if layer_sum is None:
                continue
            layer_loss = layer_sum / layer_count
        total = layer_loss if total is None else total + layer_loss
    if total is None:
        return hidden.sum() * 0.0
    return total / layers


def memory_contrastive_loss(memory, negative_memory=None, temperature=0.07, margin=0.2):
    """Optional batch memory separation loss.

    A batch cyclic shift is used as the wrong-context memory when
    ``negative_memory`` is omitted. The function is deliberately standalone so
    a later QA-conditioned positive pair can replace it without changing the
    main objective.
    """
    if memory.size(0) < 2:
        return memory.sum() * 0.0
    z=torch.nn.functional.normalize(memory.mean(1),dim=-1)
    target = z if negative_memory is None else torch.nn.functional.normalize(negative_memory.mean(1), dim=-1)
    if negative_memory is None:
        target = torch.roll(z, shifts=1, dims=0)
    positive = (z * z).sum(-1) / temperature
    negative = (z * target).sum(-1) / temperature
    return torch.relu(margin + negative - positive).mean()
def combine_losses(qa, reconstruction, qa_weight=1., reconstruction_weight=1., contrastive=None, contrastive_weight=0.):
    total=qa_weight*qa+reconstruction_weight*reconstruction
    terms={"loss":total,"qa_loss":qa,"reconstruction_loss":reconstruction}
    if contrastive is not None and contrastive_weight:
        total=total+contrastive_weight*contrastive; terms.update(contrastive_loss=contrastive, loss=total)
    return total,terms
