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
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import MetaLoRA, QuestionResampler, is_trainable_parameter_name  # noqa: E402

try:
    from transformers import Qwen3Config, Qwen3ForCausalLM
except ImportError:  # pragma: no cover - transformers is optional for the unit suite
    Qwen3Config = Qwen3ForCausalLM = None


# --- 1. the resampler in isolation -------------------------------------------------


def _resampler(B=2, Q=5, M=4, H=16, R=3, **kwargs):
    torch.manual_seed(0)
    module = QuestionResampler(H, R, num_layers=2, num_heads=4, **kwargs)
    question = torch.randn(B, Q, H)
    mask = torch.ones(B, Q, dtype=torch.bool)
    memory = torch.randn(B, M, H)
    return module, question, mask, memory


def test_readout_shape_dtype_and_parameters():
    module, question, mask, memory = _resampler()
    out = module(question, mask, memory)
    assert out.shape == (2, 3, 16)
    assert out.dtype == module.latents.dtype
    assert sum(p.numel() for p in module.parameters()) > 0
    assert torch.isfinite(out).all()
    # The read-out is injected into the backbone's input stream, so at initialisation it must
    # be at token-embedding scale (~0.02), not at LayerNorm scale (~1).
    assert 0.0 < float(out.std()) < 0.05, f"read-out scale is {float(out.std()):.3f}"
    with torch.no_grad():
        fp32 = module.to(torch.float32)
        assert fp32(question, mask, memory).dtype == torch.float32


def test_readout_bottleneck_keeps_the_parameter_count_small():
    """A full-width cross-attention block would cost ~34M per layer at H=2048."""
    torch.manual_seed(0)
    module = QuestionResampler(2048, 8, num_layers=2, num_heads=4, width=256)
    total = sum(p.numel() for p in module.parameters())
    assert total < 4_000_000, f"resampler is {total:,} parameters"
    assert total > 500_000


def test_readout_is_conditioned_on_the_question_and_the_memory():
    module, question, mask, memory = _resampler()
    base = module(question, mask, memory).detach()
    # Perturbation must not be a constant added to every feature: the resampler normalises
    # its inputs with LayerNorm, which removes exactly that component.
    other_question = question.clone()
    other_question[:, 0] += 4.0 * torch.randn_like(other_question[:, 0])
    moved_question = module(other_question, mask, memory).detach()
    assert not torch.allclose(moved_question, base), "the read-out ignores the question"
    other_memory = memory.clone()
    other_memory[:, 0] += 4.0 * torch.randn_like(other_memory[:, 0])
    moved_memory = module(question, mask, other_memory).detach()
    assert not torch.allclose(moved_memory, base), "the read-out ignores the memory"


def test_readout_survives_an_all_padding_question():
    module, question, _, memory = _resampler()
    mask = torch.zeros(2, question.size(1), dtype=torch.bool)
    out = module(question, mask, memory)
    assert torch.isfinite(out).all(), "an all-padding question must not produce NaN"


def test_readout_receives_gradient_and_ignores_masked_question_positions():
    module, question, mask, memory = _resampler(B=1)
    mask[0, -2:] = False
    out = module(question, mask, memory)
    out.sum().backward()
    assert module.latents.grad is not None and module.latents.grad.abs().sum() > 0
    # The masked positions must not influence the output (use a per-feature perturbation:
    # a constant added to every feature would also be invisible to LayerNorm).
    first = module(question, mask, memory).detach()
    question2 = question.clone()
    question2[0, -1] += 100.0 * torch.randn_like(question2[0, -1])
    second = module(question2, mask, memory).detach()
    assert torch.allclose(first, second, atol=1e-5), "a masked question position was attended"


def test_readout_rejects_bad_arguments():
    try:
        QuestionResampler(16, num_readout=0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("num_readout=0 must raise")
    try:
        QuestionResampler(16, num_readout=3, width=15, num_heads=4)
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


def _model(readout_length=0):
    torch.manual_seed(0)
    model = MetaLoRA(
        _tiny_qwen3(), rank=2, alpha=4.0, memory_length=4, decoder_hidden_size=16,
        decoder_heads=4, decoder_ffn_ratio=2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], dropout=0.0,
        max_context_tokens=16, trainable_dtype=torch.float32, context_lm=True,
        readout_length=readout_length, readout_layers=2, readout_heads=4,
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


def test_readout_off_keeps_the_historical_model():
    if Qwen3ForCausalLM is None:
        print("skip: transformers is not installed")
        return
    model = _model(readout_length=0)
    assert model.resampler is None
    output = _forward(model, _batch())
    assert output.logits.shape[:2] == (2, 4 + 5), "labels must stay [question + answer]"
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
    # The forward runs [question, read-out, answer], so the logits gain R positions -- but
    # the read-out rows are not supervised (-100), exactly like the question rows, and the
    # answer labels keep their alignment.
    assert output.logits.shape[:2] == (2, 4 + 3 + 5)
    assert output.labels.shape == (2, 4 + 3 + 5)
    assert bool((output.labels[:, 4:7] == -100).all()), "the read-out rows must not be supervised"
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


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"ok  {name}")
    print("all read-out tests passed")
