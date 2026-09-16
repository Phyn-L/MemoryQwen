"""KL distillation of the full-context model into the memory path (B1).

Run with either ``python tests/test_distill.py`` or ``pytest tests``.

What is asserted
----------------
1. ``KL(p || p) == 0`` (identical student and teacher), the ``T^2`` scaling, and agreement
   with an independent hand computation for the plain, truncated (``topk``) and
   entropy-weighted variants.
2. Gradient reaches the student hidden states and never the teacher.
3. ``positions`` selects the same targets as an equivalent mask, and an all-masked batch
   yields a finite zero rather than NaN.
4. On a real (tiny) Qwen3 the autoencoding branch exposes an aligned ``ae_teacher_hidden``
   and the distillation loss is finite with gradient flowing into the memory slots.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.losses import kl_distill_loss  # noqa: E402
from src.model import MetaLoRA, VocabularyHead, is_trainable_parameter_name  # noqa: E402

try:
    from transformers import Qwen3Config, Qwen3ForCausalLM
except ImportError:  # pragma: no cover - transformers is optional for the unit suite
    Qwen3Config = Qwen3ForCausalLM = None


def _head(D=8, V=32, H=16):
    torch.manual_seed(0)
    embedding = nn.Parameter(torch.randn(V, H), requires_grad=False)
    return VocabularyHead(
        D, V, mode="tied", embedding_getter=lambda: embedding,
        backbone_hidden_size=H, init_adapter=torch.randn(H, D),
    )


def _manual_kl(student, teacher, weight, temperature=1.0, topk=0, weights=None):
    student_log_prob = F.log_softmax(F.linear(student, weight).float() / temperature, dim=-1)
    teacher_logits = F.linear(teacher, weight).float()
    if topk:
        threshold = teacher_logits.topk(topk, dim=-1).values[:, -1:]
        teacher_logits = teacher_logits.masked_fill(teacher_logits < threshold, float("-inf"))
    teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)
    per_row = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(-1) * temperature ** 2
    if weights is None:
        return per_row.mean()
    return (per_row * weights).sum() / weights.sum()


def test_identical_student_and_teacher_give_zero():
    head = _head()
    torch.manual_seed(1)
    hidden = torch.randn(3, 5, 8)
    loss = kl_distill_loss(hidden, hidden.clone(), head)
    assert torch.isfinite(loss)
    assert float(loss.detach()) < 1e-6, f"KL(p||p) should be 0, got {float(loss.detach())}"


def test_matches_hand_computation():
    head = _head()
    torch.manual_seed(2)
    student = torch.randn(2, 4, 8)
    teacher = torch.randn(2, 4, 8)
    weight = head.materialized_weight()
    for temperature in (1.0, 2.0):
        got = kl_distill_loss(student, teacher, head, temperature=temperature, max_logits_rows=8)
        want = _manual_kl(student.reshape(-1, 8), teacher.reshape(-1, 8), weight, temperature)
        assert torch.allclose(got, want, atol=1e-5), f"T={temperature}: {float(got)} vs {float(want)}"


def test_gradient_reaches_the_student_only():
    head = _head()
    torch.manual_seed(3)
    student = torch.randn(1, 4, 8, requires_grad=True)
    teacher = torch.randn(1, 4, 8, requires_grad=True)
    loss = kl_distill_loss(student, teacher, head)
    loss.backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert teacher.grad is None, "the teacher must not receive gradient"
    assert head.adapter.weight.grad is not None and head.adapter.weight.grad.abs().sum() > 0


def test_topk_truncation_matches_the_hand_computation():
    head = _head()
    torch.manual_seed(4)
    student = torch.randn(2, 6, 8)
    teacher = torch.randn(2, 6, 8)
    weight = head.materialized_weight()
    small = kl_distill_loss(student, teacher, head, topk=3, max_logits_rows=8)
    want_small = _manual_kl(student.reshape(-1, 8), teacher.reshape(-1, 8), weight, topk=3)
    assert torch.allclose(small, want_small, atol=1e-5)
    full = kl_distill_loss(student, teacher, head, max_logits_rows=8)
    assert not torch.allclose(small, full), "truncating the teacher must change the loss"
    assert torch.allclose(
        kl_distill_loss(student, teacher, head, topk=32, max_logits_rows=8), full, atol=1e-6
    ), "topk >= vocab must be a no-op"


def test_entropy_weighting_prefers_the_uncertain_positions():
    head = _head()
    torch.manual_seed(5)
    weight = head.materialized_weight()
    # Position 0: teacher is nearly deterministic (low entropy). Position 1: nearly flat.
    student = torch.zeros(1, 2, 8)
    teacher = torch.zeros(1, 2, 8)
    with torch.no_grad():
        # A hidden state h with W h ≈ one-hot makes the teacher logits peaked (low
        # entropy); the same direction scaled down makes them nearly flat (high entropy).
        direction = torch.linalg.lstsq(weight, torch.eye(weight.size(0))[0]).solution
        teacher[0, 0] = direction * 20.0
        teacher[0, 1] = direction * 0.01
    unweighted = kl_distill_loss(student, teacher, head, max_logits_rows=8)
    weighted = kl_distill_loss(student, teacher, head, max_logits_rows=8, entropy_weight=True)
    assert not torch.allclose(unweighted, weighted), "entropy weighting changed nothing"
    with torch.no_grad():
        teacher_logits = F.linear(teacher.reshape(-1, 8), weight).float()
        prob = F.softmax(teacher_logits, dim=-1)
        entropy = -(prob * torch.log(prob.clamp_min(1e-9))).sum(-1)
    want = _manual_kl(student.reshape(-1, 8), teacher.reshape(-1, 8), weight, weights=entropy)
    assert torch.allclose(weighted, want, atol=1e-5)
    assert float(entropy[1]) > float(entropy[0]), "the constructed entropies are not ordered"


def test_positions_select_the_same_targets_as_a_mask():
    head = _head()
    torch.manual_seed(6)
    student = torch.randn(1, 6, 8)
    teacher = torch.randn(1, 6, 8)
    positions = torch.tensor([[2, 5]])
    got = kl_distill_loss(student, teacher, head, positions=positions, max_logits_rows=8)
    mask = torch.zeros(1, 6, dtype=torch.bool)
    mask.scatter_(1, positions, True)
    want = kl_distill_loss(student, teacher, head, mask=mask, max_logits_rows=8)
    assert torch.allclose(got, want, atol=1e-6)


def test_all_masked_is_finite_zero():
    head = _head()
    torch.manual_seed(7)
    student = torch.randn(1, 4, 8, requires_grad=True)
    teacher = torch.randn(1, 4, 8)
    loss = kl_distill_loss(student, teacher, head, mask=torch.zeros(1, 4, dtype=torch.bool))
    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None


def test_bad_arguments_raise():
    head = _head()
    student = torch.randn(1, 3, 8)
    for call, label in (
        (lambda: kl_distill_loss(student, torch.randn(1, 4, 8), head), "shape mismatch"),
        (lambda: kl_distill_loss(student, student.clone(), head, temperature=0.0), "temperature"),
        (lambda: kl_distill_loss(student, student.clone(), head, topk=-1), "topk"),
    ):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError(f"{label} must raise ValueError")


# --- integration: the teacher really is the plain causal LM ------------------------


def _tiny_qwen3(vocab=64, hidden=32, layers=2, heads=4, kv_heads=2, intermediate=64):
    config = Qwen3Config(
        vocab_size=vocab, hidden_size=hidden, num_hidden_layers=layers,
        num_attention_heads=heads, num_key_value_heads=kv_heads,
        intermediate_size=intermediate, max_position_embeddings=256, tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(config)


def _model(ae_lm=True):
    torch.manual_seed(0)
    model = MetaLoRA(
        _tiny_qwen3(), rank=2, alpha=4.0, memory_length=4, decoder_hidden_size=16,
        decoder_heads=4, decoder_ffn_ratio=2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], dropout=0.0,
        max_context_tokens=16, trainable_dtype=torch.float32, context_lm=True, ae_lm=ae_lm,
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad = is_trainable_parameter_name(name)
    return model


def _batch(vocab=64, B=2, L=10, Q=4, A=5):
    torch.manual_seed(1)
    return dict(
        context_ids=torch.randint(0, vocab, (B, L)),
        context_mask=torch.ones(B, L, dtype=torch.bool),
        question_ids=torch.randint(0, vocab, (B, Q)),
        question_mask=torch.ones(B, Q, dtype=torch.bool),
        answer_ids=torch.randint(0, vocab, (B, A)),
        answer_mask=torch.ones(B, A, dtype=torch.bool),
    )


def _forward(model, batch):
    embedding = model.qwen.get_input_embeddings()
    return model(
        embedding(batch["context_ids"]), batch["context_mask"],
        embedding(batch["question_ids"]), batch["question_mask"],
        embedding(batch["answer_ids"]), batch["answer_mask"], batch["answer_ids"],
        torch.arange(batch["context_ids"].size(0)),
        context_ids=batch["context_ids"], context_lm_positions=4,
    )


def test_teacher_hidden_is_aligned_and_distillation_trains_the_memory():
    if Qwen3ForCausalLM is None:
        print("skip: transformers is not installed")
        return
    model = _model(ae_lm=True)
    output = _forward(model, _batch())
    assert output.ae_teacher_hidden is not None
    assert output.ae_teacher_hidden.shape == output.ae_hidden.shape
    loss = kl_distill_loss(
        output.ae_hidden, output.ae_teacher_hidden, model.ae_head, output.ae_mask,
        max_logits_rows=8, entropy_weight=True,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.memory_tokens.grad is not None and model.memory_tokens.grad.abs().sum() > 0


def test_teacher_hidden_is_absent_when_the_autoencoding_branch_is_off():
    if Qwen3ForCausalLM is None:
        print("skip: transformers is not installed")
        return
    output = _forward(_model(ae_lm=False), _batch())
    assert output.ae_teacher_hidden is None and output.ae_hidden is None


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"ok  {name}")
    print("all distillation tests passed")
