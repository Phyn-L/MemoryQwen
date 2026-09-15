import torch
import torch.nn.functional as F

def qa_loss(logits, labels):
    token_logits = logits[..., :-1, :]
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
