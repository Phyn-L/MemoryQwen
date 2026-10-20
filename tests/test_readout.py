"""Question-conditioned read-out over the cached memory (C3).

Run with either ``python tests/test_readout.py`` or ``pytest tests``.

What is asserted
----------------
1. ``QuestionResampler`` returns ``[B, R, H]`` in the module dtype, is conditioned on both
   the question and the memory, survives an all-padding question, and receives gradient.
2. With ``readout_length=0`` (the default) the model has no resampler at all, so the
   historical path is untouched; the mask rule itself is pinned in ``tests/test_masks.py``.
3. With the read-out on, a real (tiny) Qwen3 forward still produces answer-aligned logits,
   the resampler's parameters receive gradient from the answer loss (i.e. the answer really
   attends to the read-out), and greedy generation runs and calls the resampler.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import ContextMemoryModel, QuestionResampler, is_trainable_parameter_name  # noqa: E402

try:
    from transformers import Qwen3Config, Qwen3ForCausalLM
except ImportError:  # pragma: no cover - transformers is optional for the unit suite
    Qwen3Config = Qwen3ForCausalLM = None


# --- 1. the resampler in isolation -------------------------------------------------


def _resampler(B=2, Q=5, M=4, H=16, R=3, **kwargs):
    torch.manual_seed(0)
    module = QuestionResampler(H, question_resampler_length=R,
                               question_resampler_layers=2,
                               question_resampler_heads=4, **kwargs)
    hidden = torch.randn(B, M + Q, H)
    mask = torch.ones(B, M + Q, dtype=torch.bool)
    return module, hidden, mask


def test_readout_shape_dtype_and_parameters():
    module, hidden, mask = _resampler()
    out = module(hidden, mask)
    assert out.shape == (2, 3, 16)
    assert out.dtype == module.latents.dtype
    assert sum(p.numel() for p in module.parameters()) > 0
    assert torch.isfinite(out).all()
    # The read-out is injected into the backbone's input stream, so at initialisation it must
    # be at token-embedding scale (~0.02), not at LayerNorm scale (~1).
    assert 0.0 < float(out.std()) < 0.05, f"read-out scale is {float(out.std()):.3f}"
    with torch.no_grad():
        fp32 = module.to(torch.float32)
        assert fp32(hidden, mask).dtype == torch.float32


def test_readout_bottleneck_keeps_the_parameter_count_small():
    """A full-width cross-attention block would cost ~34M per layer at H=2048."""
    torch.manual_seed(0)
    module = QuestionResampler(
        2048, question_resampler_length=8, question_resampler_layers=2,
        question_resampler_heads=4, question_resampler_hidden_size=256,
    )
    total = sum(p.numel() for p in module.parameters())
    assert total < 4_000_000, f"resampler is {total:,} parameters"
    assert total > 500_000


def test_readout_keeps_its_own_dtype_under_bf16_autocast():
    """Regression: accelerate runs the whole training forward under bf16 autocast.

    A float32 Linear still returns bfloat16 under autocast, so a projection computed
    outside ``no_autocast`` met the float32 LayerNorms and raised "expected scalar type
    BFloat16 but found Float". The module owns its dtype: everything inside stays float32.
    """
    module, hidden, mask = _resampler()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = module(hidden, mask)
    assert out.dtype == torch.float32, f"read-out came back {out.dtype} under autocast"
    assert torch.isfinite(out).all()
    # and the value is the same as with autocast off, because nothing was downcast
    with torch.no_grad():
        plain = module(hidden, mask)
    assert torch.allclose(out.detach(), plain, atol=1e-5)


def test_readout_is_conditioned_on_the_question_and_the_memory():
    module, hidden, mask = _resampler()
    base = module(hidden, mask).detach()
    # Perturbation must not be a constant added to every feature: the resampler normalises
    # its inputs with LayerNorm, which removes exactly that component.
    changed_question = hidden.clone()
    changed_question[:, -1] += 4.0 * torch.randn_like(changed_question[:, -1])
    assert not torch.allclose(module(changed_question, mask), base), "the read-out ignores question hidden"
    changed_memory = hidden.clone()
    changed_memory[:, 0] += 4.0 * torch.randn_like(changed_memory[:, 0])
    assert not torch.allclose(module(changed_memory, mask), base), "the read-out ignores memory hidden"


def test_readout_survives_an_all_padding_question():
    module, hidden, _ = _resampler()
    mask = torch.ones_like(hidden[:, :, 0], dtype=torch.bool)
    mask[:, -5:] = False
    out = module(hidden, mask)
    assert torch.isfinite(out).all(), "an all-padding question must not produce NaN"


def test_readout_receives_gradient_and_ignores_masked_question_positions():
    module, hidden, mask = _resampler(B=1)
    mask[0, -2:] = False
    out = module(hidden, mask)
    out.sum().backward()
    assert module.latents.grad is not None and module.latents.grad.abs().sum() > 0
    # The masked positions must not influence the output (use a per-feature perturbation:
    # a constant added to every feature would also be invisible to LayerNorm).
    first = module(hidden, mask).detach()
    hidden2 = hidden.clone()
    hidden2[0, -1] += 100.0 * torch.randn_like(hidden2[0, -1])
    second = module(hidden2, mask).detach()
    assert torch.allclose(first, second, atol=1e-5), "a masked hidden position was attended"


def test_readout_rejects_bad_arguments():
    try:
        QuestionResampler(16, question_resampler_length=0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("num_readout=0 must raise")
    try:
        QuestionResampler(16, question_resampler_length=3,
                          question_resampler_hidden_size=15,
                          question_resampler_heads=4)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("width not divisible by num_heads must raise")


# --- 2. model integration ---------------------------------------------------------


def _tiny_qwen3(vocab=64, hidden=32, layers=2, heads=4, kv_heads=2, intermediate=64):
    config = Qwen3Config(
        vocab_size=vocab, hidden_size=hidden, num_hidden_layers=layers,
        num_attention_heads=heads, num_key_value_heads=kv_heads,
        intermediate_size=intermediate, max_position_embeddings=256, tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(config)


def _model(readout_length=0, readout_mode="memory_question"):
    torch.manual_seed(0)
    model = ContextMemoryModel(
        _tiny_qwen3(), rank=2, alpha=4.0, memory_length=4,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], dropout=0.0,
        max_context_tokens=16, trainable_dtype=torch.float32,
        readout_length=readout_length, question_resampler_layers=2,
        question_resampler_heads=4, question_resampler_ffn_ratio=2,
        readout_mode=readout_mode,
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


def _forward(model, batch, **kwargs):
    embedding = model.qwen.get_input_embeddings()
    return model(
        embedding(batch["context_ids"]), batch["context_mask"],
        embedding(batch["question_ids"]), batch["question_mask"],
        embedding(batch["answer_ids"]), batch["answer_mask"], batch["answer_ids"],
        torch.arange(batch["context_ids"].size(0)),
        context_ids=batch["context_ids"],
        **kwargs,
    )


def test_readout_off_keeps_the_historical_model():
    if Qwen3ForCausalLM is None:
        print("skip: transformers is not installed")
        return
    model = _model(readout_length=0)
    assert model.resampler is None
    output = _forward(model, _batch())
    assert output.logits.shape[:2] == (2, 4 + 5), "QA has no task embedding"
    assert output.labels.shape == (2, 9)
    # the resampler name is still wired into the trainable-parameter rule
    assert is_trainable_parameter_name("resampler.latents")


def test_readout_is_conditioned_and_trains_the_resampler():
    if Qwen3ForCausalLM is None:
        print("skip: transformers is not installed")
        return
    model = _model(readout_length=3)
    assert model.resampler is not None
    batch = _batch()
    output = _forward(model, batch)
    assert output.logits.shape[:2] == (2, 4 + 3 + 5)
    assert output.labels.shape == (2, 12)
    assert bool((output.labels[:, :7] == -100).all())
    assert torch.equal(output.labels[:, 7:], batch["answer_ids"])
    loss = output.logits.float().logsumexp(-1).mean()      # any differentiable scalar
    loss.backward()
    grads = [p.grad for p in model.resampler.parameters()]
    assert all(g is not None for g in grads), "the resampler got no gradient at all"
    assert any(g.abs().sum() > 0 for g in grads), (
        "the answer rows do not attend to the read-out (mask or position wiring is wrong)"
    )


def test_generation_calls_the_resampler_and_returns_tokens():
    if Qwen3ForCausalLM is None:
        print("skip: transformers is not installed")
        return
    model = _model(readout_length=3)
    model.eval()
    batch = _batch()
    embedding = model.qwen.get_input_embeddings()
    prefix = model.encode_context_prefix(embedding(batch["context_ids"]), batch["context_mask"])

    calls = {"n": 0}
    original = model.resampler.forward

    def counting_forward(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    model.resampler.forward = counting_forward

    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = None

    generated = model.generate_answers_with_prefix(
        prefix, torch.arange(batch["context_ids"].size(0)),
        batch["question_ids"], batch["question_mask"], _Tokenizer(), max_new_tokens=4,
    )
    assert calls["n"] > 0, "generation never called the resampler"
    assert generated.shape[0] == batch["context_ids"].size(0)
    assert generated.shape[1] == 4


def test_generation_without_readout_still_works():
    if Qwen3ForCausalLM is None:
        print("skip: transformers is not installed")
        return
    model = _model(readout_length=0)
    model.eval()
    batch = _batch()
    embedding = model.qwen.get_input_embeddings()
    prefix = model.encode_context_prefix(embedding(batch["context_ids"]), batch["context_mask"])

    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = None

    generated = model.generate_answers_with_prefix(
        prefix, torch.arange(batch["context_ids"].size(0)),
        batch["question_ids"], batch["question_mask"], _Tokenizer(), max_new_tokens=4,
    )
    assert generated.shape == (2, 4)




def test_memory_only_input_dependency():
    module, hidden, mask = _resampler(mode="memory_only")
    hidden.requires_grad_()
    out = module(hidden, mask)
    assert not torch.allclose(out, module(hidden + torch.randn_like(hidden), mask))
    out.square().sum().backward()
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0


def test_new_modes_forward_gradient_and_generation():
    for mode in ("memory_only", "learned_prefix", "off"):
        model = _model(3, mode)
        batch = _batch()
        output = _forward(model, batch)
        extra = 0 if mode == "off" else 3
        assert output.logits.shape[1] == 9 + extra
        assert torch.equal(output.labels[:, 4 + extra:], batch["answer_ids"])
        if extra:
            assert (output.labels[:, 4:7] == -100).all()
        # The training path is the only loss path now: per-row means over the memory-prefixed
        # forward, with no [rows, tokens, vocab] logits materialised.
        graded = _forward(model, batch, loss_only=True)
        assert graded.supervision_losses.shape == (output.labels.size(0),)
        assert bool(graded.supervision_mask.all())
        graded.supervision_losses.sum().backward()
        if mode == "off":
            assert model.resampler is None and model.readout_length == 0
        else:
            assert all(p.grad is not None for p in model.resampler.parameters())
            assert any(p.grad.abs().sum() > 0 for p in model.resampler.parameters())
            assert all(
                is_trainable_parameter_name(name)
                for name, _ in model.named_parameters()
                if name.startswith("resampler.")
            )
        model.eval()
        embed = model.qwen.get_input_embeddings()
        with torch.no_grad():
            prefix = model.encode_context_prefix(embed(batch["context_ids"]), batch["context_mask"])
            if mode == "learned_prefix":
                a = model._readout_embeddings(
                    prefix, embed(batch["question_ids"]), batch["question_mask"]
                )
                b = model._readout_embeddings(
                    prefix, torch.randn_like(embed(batch["question_ids"])), batch["question_mask"]
                )
                assert torch.allclose(a, b)
            class Tokenizer:
                pad_token_id = 0
                eos_token_id = None
            tokens = model.generate_answers_with_prefix(
                prefix, torch.arange(2), batch["question_ids"], batch["question_mask"],
                Tokenizer(), max_new_tokens=3,
            )
            assert tokens.shape == (2, 3)


def test_readout_can_remove_question_from_answer_input():
    model = _model(3)
    model.readout_keep_question_tokens = False
    batch = _batch()
    output = _forward(model, batch)
    assert output.logits.shape[1] == 3 + 5
    assert torch.equal(output.labels[:, 3:], batch["answer_ids"])
    assert (output.labels[:, :3] == -100).all()
    model.eval()
    embedding = model.qwen.get_input_embeddings()
    with torch.no_grad():
        prefix = model.encode_context_prefix(embedding(batch["context_ids"]), batch["context_mask"])
        class Tokenizer:
            pad_token_id = 0
            eos_token_id = None
        generated = model.generate_answers_with_prefix(
            prefix, torch.arange(2), batch["question_ids"], batch["question_mask"],
            Tokenizer(), max_new_tokens=3,
        )
    assert generated.shape == (2, 3)


def test_readout_mode_config_validation():
    from utils.config import TrainConfig
    for mode in ("memory_question", "memory_only", "learned_prefix", "off"):
        cfg = TrainConfig()
        cfg.memory.readout_mode = mode
        cfg.memory.readout_length = 3
        cfg.validate()
    cfg.memory.readout_mode = "typo"
    import pytest
    with pytest.raises(ValueError, match="readout_mode"):
        cfg.validate()


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"ok  {name}")
    print("all read-out tests passed")
