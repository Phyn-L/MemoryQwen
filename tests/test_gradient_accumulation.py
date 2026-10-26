"""Gradient accumulation: one optimizer step averages its micro-batches.

The training loop divides the step budget and the loss by
``training.gradient_accumulation_steps`` and only steps the optimizer on the group boundary.
These tests pin the two halves that are easy to get wrong: the arithmetic (an accumulated group
must produce the same gradient as one combined batch) and the step/skip bookkeeping (the
scheduler, the evaluation cadence and ``total_steps`` count optimizer steps, while the loop
indexes micro-batches).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.training.resume import micro_batch_skip, optimizer_steps  # noqa: E402
from src.training.stages import sample_task_masks  # noqa: E402
from utils.config import LossConfig, TrainConfig  # noqa: E402


def _gradients(model, batches, accumulation):
    model.zero_grad(set_to_none=True)
    for inputs, targets in batches:
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        (loss / accumulation).backward()
    return torch.cat([parameter.grad.detach().reshape(-1) for parameter in model.parameters()])


def test_an_accumulated_group_produces_the_same_gradient_as_one_batch():
    torch.manual_seed(0)
    inputs = torch.randn(6, 4)
    targets = torch.randn(6, 1)

    torch.manual_seed(1)
    combined_model = torch.nn.Linear(4, 1)
    torch.manual_seed(1)
    accumulated_model = torch.nn.Linear(4, 1)

    combined = _gradients(combined_model, [(inputs, targets)], 1)
    split = [
        (inputs[index:index + 2], targets[index:index + 2]) for index in range(0, 6, 2)
    ]
    accumulated = _gradients(accumulated_model, split, 3)
    torch.testing.assert_close(combined, accumulated)


def test_a_group_shorter_than_the_accumulation_steps_is_still_an_average():
    torch.manual_seed(2)
    model = torch.nn.Linear(3, 1)
    inputs, targets = torch.randn(1, 3), torch.randn(1, 1)
    # The last group of an epoch can be shorter than `accumulation`; the loop divides by the
    # configured accumulation, exactly like accelerate's backward() does.
    short = _gradients(model, [(inputs, targets)], 4)
    single_model = torch.nn.Linear(3, 1)
    single_model.load_state_dict(model.state_dict())
    plain = _gradients(single_model, [(inputs, targets)], 1)
    torch.testing.assert_close(short * 4, plain)


def test_optimizer_steps_round_up_and_skip_whole_groups():
    assert optimizer_steps(8, 1) == 8
    assert optimizer_steps(8, 4) == 2
    assert optimizer_steps(9, 4) == 3, "a partial last group still steps"
    assert optimizer_steps(0, 4) == 0

    # 3 optimizer steps per epoch with 4 micro-batches each: resuming at step 5 means
    # epoch 1 has already consumed 2 groups = 8 micro-batches.
    assert micro_batch_skip(5, 3, 4) == 8
    assert micro_batch_skip(3, 3, 4) == 0
    assert micro_batch_skip(0, 3, 4) == 0
    assert micro_batch_skip(5, 0, 4) == 0


def test_the_config_validates_the_accumulation_and_defaults_to_one_step_per_batch():
    assert TrainConfig().training.gradient_accumulation_steps == 1
    with_accumulation = TrainConfig.from_dict({"training": {"gradient_accumulation_steps": 8}})
    with_accumulation.validate()
    assert with_accumulation.training.gradient_accumulation_steps == 8
    for bad in (0, -1):
        cfg = TrainConfig.from_dict({"training": {"gradient_accumulation_steps": bad}})
        try:
            cfg.validate()
        except ValueError as error:
            assert "gradient_accumulation_steps" in str(error)
        else:
            raise AssertionError("a non-positive accumulation must be rejected")


def test_the_objective_weights_are_scheduled_per_optimizer_step():
    from src.losses import scheduled_task_weights

    cfg = TrainConfig.from_dict(
        {
            "data": {
                "train_stage": "mix",
                "recon_schedule": [0.5, 0.5],
                "qa_schedule": [0.5, 0.5],
            }
        }
    )
    assert scheduled_task_weights(0, 100, cfg.data) == (0.5, 0.5)

    independent = TrainConfig.from_dict(
        {
            "data": {
                "train_stage": "mix",
                "recon_schedule": [0.2, 0.3],
                "qa_schedule": [0.4, 0.5],
            }
        }
    )
    independent.validate()


def test_independent_task_sampling_keeps_every_context_and_allows_both_tasks():
    qa_contexts = torch.tensor([0, 0, 1, 1, 2, 2])
    recon_contexts = torch.tensor([0, 1, 2])
    qa_rows, recon_rows = sample_task_masks(
        qa_contexts, recon_contexts, 0.4, 0.3, seed=7, num_contexts=3
    )
    assert torch.all(qa_rows | recon_rows[qa_contexts])
    assert torch.equal(
        qa_rows,
        sample_task_masks(
            qa_contexts, recon_contexts, 0.4, 0.3, seed=7, num_contexts=3
        )[0],
    )


def test_task_sampling_falls_back_when_both_probabilities_are_zero():
    qa_rows, recon_rows = sample_task_masks(
        torch.arange(4), torch.arange(4), 0.0, 0.0, seed=1, num_contexts=4
    )
    assert torch.all(qa_rows | recon_rows)
    assert LossConfig().qa_weight == 1.0
