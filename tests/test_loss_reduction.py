"""The objective's reduction contract.

Two properties decide whether the reported losses mean what the config says:

1. every supervision row is averaged over *its own* active tokens before the rows of a task are
   averaged, so a long Recon context cannot outweigh a short QA answer;
2. that happens in one flattened vocabulary projection, so the chunk size
   (``loss.max_logits_rows``) must not change any number.

The distillation position sampler is pinned here too: with an explicit generator it has to be
recomputable from (seed, step), which is what lets a resumed run continue the *same* run.
"""
from __future__ import annotations

import random

import torch
from torch import nn

from src.losses import (
    sample_positions,
    sequence_lm_loss,
    supervision_objectives,
    teacher_forced_row_stats,
)
from utils.config import LossConfig


class _Head(nn.Module):
    """A vocabulary head shaped like the tied unembedding ``[vocab, hidden]``."""

    def __init__(self, vocab: int = 16, hidden: int = 8):
        super().__init__()
        generator = torch.Generator().manual_seed(vocab * 100 + hidden)
        self.weight = nn.Parameter(
            torch.randn(vocab, hidden, generator=generator) / hidden ** 0.5
        )


def _pair(rows=2, tokens=5, vocab=16, hidden=8, seed=0):
    torch.manual_seed(seed)
    return torch.randn(rows, tokens, hidden), torch.randint(0, vocab, (rows, tokens))


def test_row_means_do_not_depend_on_the_other_rows_token_counts():
    head = _Head()
    hidden, labels = _pair()
    labels[0, 1:] = -100            # one active token
    row_losses, row_mask = sequence_lm_loss(hidden, labels, head, max_logits_rows=3)
    assert row_mask.tolist() == [True, True]

    # Reference: the same numbers, one row at a time (the shape the loss had before).
    reference = torch.stack([
        sequence_lm_loss(hidden[index:index + 1], labels[index:index + 1], head)[0][0]
        for index in range(hidden.size(0))
    ])
    torch.testing.assert_close(row_losses, reference)

    # And the task mean is the mean of the two row means -- not the mean over all six tokens.
    token_weighted = (row_losses[0] * 1 + row_losses[1] * 5) / 6
    assert not torch.allclose(row_losses.mean(), token_weighted)


def test_the_logits_chunk_size_does_not_change_the_loss():
    head = _Head()
    hidden, labels = _pair(rows=3, tokens=4)
    labels[0, 2:] = -100
    labels[2, 0] = -100
    reference, reference_mask = sequence_lm_loss(hidden, labels, head, max_logits_rows=1)
    for chunk in (2, 3, 7, 64):
        got, got_mask = sequence_lm_loss(hidden, labels, head, max_logits_rows=chunk)
        torch.testing.assert_close(got, reference)
        assert torch.equal(got_mask, reference_mask)


def test_a_row_without_targets_is_kept_out_of_the_task_mean():
    head = _Head()
    hidden, labels = _pair(rows=3, tokens=3)
    labels[0, :] = -100
    labels[2, :] = -100
    losses, mask = sequence_lm_loss(hidden, labels, head)
    assert mask.tolist() == [False, True, False]
    assert losses[0].item() == 0.0 and losses[2].item() == 0.0
    total, terms = supervision_objectives(
        losses, torch.ones(3, dtype=torch.long), LossConfig(), row_mask=mask
    )
    torch.testing.assert_close(terms["qa_loss"], losses[1])
    torch.testing.assert_close(total, losses[1])
    # Without the mask the two empty rows would drag the mean down by a factor of three.
    unmasked, _ = supervision_objectives(
        losses, torch.ones(3, dtype=torch.long), LossConfig()
    )
    torch.testing.assert_close(unmasked, losses.mean())


def test_gradients_reach_every_row_of_the_flattened_projection():
    head = _Head()
    hidden, labels = _pair(rows=4, tokens=3)
    hidden = hidden.clone().requires_grad_(True)
    losses, mask = sequence_lm_loss(hidden, labels, head, max_logits_rows=2)
    assert bool(mask.all())
    losses.sum().backward()
    assert hidden.grad is not None
    assert bool(torch.isfinite(hidden.grad).all())
    assert hidden.grad.abs().sum() > 0


def test_sample_positions_is_reproducible_from_an_explicit_generator():
    mask = torch.ones(2, 8, dtype=torch.bool)

    def draw(seed):
        return sample_positions(mask, 3, generator=torch.Generator().manual_seed(seed))

    first, keep = draw(42)
    again, _ = draw(42)
    assert torch.equal(first, again)
    assert bool(keep.all())
    other, _ = draw(43)
    assert not torch.equal(first, other)

    # A short row leaves -1 in the slots that could not be filled, so a caller that forgot the
    # keep-mask cannot silently score another token instead.
    short = torch.zeros(1, 8, dtype=torch.bool)
    short[0, :2] = True
    positions, keep = sample_positions(short, 4, generator=torch.Generator().manual_seed(0))
    assert int(keep.sum()) == 2
    assert bool((positions[0][~keep[0]] == -1).all())


def test_sample_positions_accepts_a_step_derived_generator_instead_of_the_global_stream():
    """Same (seed, step) -> same subset, and consuming the global RNG changes nothing."""
    mask = torch.ones(1, 16, dtype=torch.bool)
    generator = torch.Generator().manual_seed(7 + 3)
    first, _ = sample_positions(mask, 4, generator=generator)
    random.random()
    torch.rand(99)
    generator.manual_seed(7 + 3)
    again, _ = sample_positions(mask, 4, generator=generator)
    assert torch.equal(first, again)


def test_teacher_forced_row_stats_match_a_manual_computation():
    head = _Head()
    hidden, labels = _pair(rows=2, tokens=3)
    labels[1, 2] = -100
    stats = teacher_forced_row_stats(hidden, labels, head)
    logits = hidden @ head.weight.t()
    log_prob = torch.log_softmax(logits.float(), dim=-1)
    safe = labels.clamp_min(0)
    nll = -log_prob.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    nll = nll.masked_fill(labels.eq(-100), 0.0)
    torch.testing.assert_close(stats["nll"], nll.sum(dim=1))
    torch.testing.assert_close(stats["tokens"], labels.ne(-100).sum(dim=1).float())
    correct = logits.argmax(-1).eq(labels).masked_fill(labels.eq(-100), False)
    torch.testing.assert_close(stats["correct"], correct.sum(dim=1).float())
