"""Unit tests for the optional full-context KL distillation objective."""
from __future__ import annotations

import torch
from torch import nn

from src.losses import kl_distill_loss, memory_objectives
from src.model import ContextMemoryModel, is_trainable_parameter_name
from utils.config import LossConfig

try:
    from transformers import Qwen3Config, Qwen3ForCausalLM
except ImportError:  # pragma: no cover
    Qwen3Config = Qwen3ForCausalLM = None


def _head(hidden=8, vocab=32):
    torch.manual_seed(0)
    return nn.Linear(hidden, vocab, bias=False)


def _tiny_model():
    config = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    model = ContextMemoryModel(
        Qwen3ForCausalLM(config), rank=2, alpha=4.0, memory_length=4,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        max_context_tokens=16, distillation=True,
        recon_user_token_ids=(2,), recon_answer_prefix_token_ids=(3, 4),
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad = is_trainable_parameter_name(name)
    return model


def test_identical_student_and_teacher_give_zero():
    head = _head()
    hidden = torch.randn(2, 4, 8)
    loss = kl_distill_loss(hidden, hidden.clone(), head)
    assert torch.isfinite(loss)
    assert float(loss) < 1e-6


def test_gradient_reaches_student_only():
    head = _head()
    student = torch.randn(1, 4, 8, requires_grad=True)
    teacher = torch.randn(1, 4, 8, requires_grad=True)
    kl_distill_loss(student, teacher, head).backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert teacher.grad is None
    assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0


def test_positions_match_mask():
    head = _head()
    student = torch.randn(1, 6, 8)
    teacher = torch.randn(1, 6, 8)
    positions = torch.tensor([[2, 5]])
    mask = torch.zeros(1, 6, dtype=torch.bool)
    mask.scatter_(1, positions, True)
    assert torch.allclose(
        kl_distill_loss(student, teacher, head, positions=positions),
        kl_distill_loss(student, teacher, head, mask=mask),
    )


def test_model_distillation_uses_memory_gradient():
    if Qwen3ForCausalLM is None:
        return
    model = _tiny_model()
    ids = torch.randint(0, 64, (2, 8))
    mask = torch.ones_like(ids, dtype=torch.bool)
    prefix = model.encode_context_prefix(model.qwen.get_input_embeddings()(ids), mask)
    cfg = LossConfig(distill_weight=1.0, distill_positions=4)
    terms = memory_objectives(model, prefix, ids, mask, cfg)
    assert set(terms) == {"distill_loss"}
    terms["distill_loss"].backward()
    assert model.context_resampler.latents.grad is not None
    assert model.context_resampler.latents.grad.abs().sum() > 0
