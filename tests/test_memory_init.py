"""Memory slot initialisation and slot attention (C2).

Run with either ``python tests/test_memory_init.py`` or ``pytest tests``.

What is asserted
----------------
1. The default is still the historical ``randn * 0.02`` (so nothing changes silently)
   and no init mode touches the backbone or the dtypes.
2. ``token_embed`` really draws distinct rows of the frozen token embedding, is
   reproducible from ``init_seed``, and survives the trainable-dtype cast.
3. ``vocab_mean`` starts at the embedding mean with the documented 0.02 noise.
4. ``allow_slot_attention`` defaults to False and is recorded on the model; the mask
   rule it turns on is pinned in ``tests/test_masks.py`` (independent oracle).
"""
from __future__ import annotations

from types import SimpleNamespace
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import MetaLoRA  # noqa: E402


class _FakeQwen(nn.Module):
    """Minimal stand-in for a Qwen backbone (no transformers needed)."""

    def __init__(self, dim=16, layers=2, vocab=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(
            [nn.ModuleDict({"q_proj": nn.Linear(dim, dim, bias=False)}) for _ in range(layers)]
        )
        self.config = SimpleNamespace(hidden_size=dim, num_hidden_layers=layers)

    def get_input_embeddings(self):
        return self.embed


def _model(**kwargs):
    torch.manual_seed(0)
    options = dict(
        rank=2,
        alpha=4.0,
        memory_length=4,
        decoder_hidden_size=8,
        decoder_heads=2,
        decoder_ffn_ratio=2,
        target_modules=["q_proj"],
        max_context_tokens=8,
        trainable_dtype=torch.float32,
        context_lm=False,
    )
    options.update(kwargs)
    return MetaLoRA(_FakeQwen(), **options)


def _is_embedding_row(slot, embedding):
    return bool((slot.unsqueeze(0) == embedding).all(dim=-1).any())


def test_default_init_is_the_historical_randn():
    model = _model()
    assert model.init_mode == "randn"
    embedding = model.qwen.get_input_embeddings().weight
    assert not _is_embedding_row(model.memory_tokens[0], embedding), (
        "the default init must not be a real token embedding"
    )
    assert model.memory_tokens.dtype == torch.float32
    assert model.memory_tokens.shape == (4, 16)


def test_token_embed_draws_real_embedding_rows():
    model = _model(init_mode="token_embed", init_seed=3)
    embedding = model.qwen.get_input_embeddings().weight
    for slot in model.memory_tokens:
        assert _is_embedding_row(slot, embedding), "slot is not a vocabulary embedding"
    rows = {tuple(row.tolist()) for row in model.memory_tokens}
    assert len(rows) == model.memory_tokens.size(0), "slots must be distinct"
    assert torch.isfinite(model.memory_tokens).all()


def test_token_embed_is_reproducible_from_init_seed():
    first = _model(init_mode="token_embed", init_seed=7)
    second = _model(init_mode="token_embed", init_seed=7)
    other = _model(init_mode="token_embed", init_seed=8)
    assert torch.equal(first.memory_tokens, second.memory_tokens)
    assert not torch.equal(first.memory_tokens, other.memory_tokens)


def test_vocab_mean_starts_at_the_embedding_mean():
    model = _model(init_mode="vocab_mean", init_seed=1)
    embedding = model.qwen.get_input_embeddings().weight
    mean = embedding.mean(dim=0)
    deviation = (model.memory_tokens - mean).abs().max()
    assert float(deviation) < 0.15, f"vocab_mean is {float(deviation):.3f} away from the mean"
    assert float((model.memory_tokens - mean).std()) > 0.0


def test_init_modes_produce_different_slots():
    randn = _model()
    tokens = _model(init_mode="token_embed")
    mean = _model(init_mode="vocab_mean")
    assert not torch.allclose(randn.memory_tokens, tokens.memory_tokens)
    assert not torch.allclose(randn.memory_tokens, mean.memory_tokens)


def test_token_embed_respects_the_trainable_dtype():
    model = _model(init_mode="token_embed", trainable_dtype=torch.bfloat16)
    assert model.memory_tokens.dtype == torch.bfloat16
    assert model.qwen.get_input_embeddings().weight.dtype == torch.float32


def test_unknown_init_mode_raises():
    try:
        _model(init_mode="bogus")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("an unknown init_mode must raise")


def test_slot_attention_defaults_off_and_is_recorded():
    assert _model().allow_slot_attention is False
    assert _model(allow_slot_attention=True).allow_slot_attention is True


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"ok  {name}")
    print("all memory-init tests passed")
