"""The objective's reduction contract.

Two properties decide whether the reported losses mean what the config says:

1. every supervision row is averaged over *its own* active tokens before the rows of a task are
   averaged, so a long Recon context cannot outweigh a short QA answer;
2. that happens in one flattened vocabulary projection, so the chunk size
   (``loss.max_logits_rows``) must not change any number.

"""
from __future__ import annotations

import torch
from torch import nn

from src.losses import (
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


def test_teacher_forced_stats_project_once_per_chunk(monkeypatch):
    import torch.nn.functional as functional

    head = _Head()
    hidden, labels = _pair(rows=2, tokens=3)
    calls = []
    original = functional.linear

    def counted_linear(input, weight, bias=None):
        calls.append(weight)
        return original(input, weight, bias)

    monkeypatch.setattr("src.losses.F.linear", counted_linear)
    teacher_forced_row_stats(hidden, labels, head, max_logits_rows=2)
    assert len(calls) == 3
