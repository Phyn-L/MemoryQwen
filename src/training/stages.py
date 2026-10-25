from __future__ import annotations

import torch


def training_stage(train_stage: str, epoch: int, epochs_per_stage: int) -> str:
    if train_stage == "two_stage":
        return "recon" if epoch < epochs_per_stage else "qa"
    return train_stage


def sample_task_masks(
    qa_context_indices: torch.Tensor,
    recon_context_indices: torch.Tensor,
    qa_probability: float,
    recon_probability: float,
    seed: int,
    num_contexts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample independent task flags and force one flag for every context."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    qa = torch.rand(num_contexts, generator=generator) < qa_probability
    recon = torch.rand(num_contexts, generator=generator) < recon_probability
    missing = ~(qa | recon)
    if bool(missing.any()):
        qa_only = qa_probability * (1.0 - recon_probability)
        recon_only = recon_probability * (1.0 - qa_probability)
        denominator = qa_only + recon_only
        fallback_qa = 0.5 if denominator <= 0 else qa_only / denominator
        fallback = torch.rand(num_contexts, generator=generator) < fallback_qa
        qa[missing] = fallback[missing]
        recon[missing] = ~fallback[missing]
    return qa[qa_context_indices.cpu()].to(qa_context_indices.device), recon[
        recon_context_indices.cpu()
    ].to(recon_context_indices.device)


