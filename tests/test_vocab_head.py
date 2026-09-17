"""Tied vocabulary head (A1).

Run with either ``python tests/test_vocab_head.py`` or ``pytest tests``.

What is asserted
----------------
1. ``VocabularyHead(mode="linear")`` is indistinguishable from the ``nn.Linear`` it
   replaces: same parameter name, same shape, same forward value, and a default
   ``MetaLoRA`` still exposes ``context_lm_head.weight`` so old checkpoints load.
2. ``mode="tied"`` really scores with the frozen embedding (``W == E @ adapter``),
   keeps the per-row cost at ``D*vocab`` by materialising ``W``, and trains only the
   ``D -> H`` adapter: gradient reaches the adapter, never ``E``.
3. The materialised weight is invalidated by an in-place parameter update (the
   optimizer step), so a stale head cannot score a later forward.
4. ``head_init="memory_projection"`` starts from the decoders' memory projection, so
   the step-0 logits already score the tokens the memory points at.
5. ``context_lm_loss`` runs end to end with a tied head and backpropagates into it.
6. A materialisation performed under ``torch.no_grad()`` (every evaluation) never becomes
   the cached weight a later training step scores with -- that is what made the ON arm die
   on the first step after its first validation with DDP's "did not receive grad" error.
"""
from __future__ import annotations

from types import SimpleNamespace
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.losses import context_lm_loss  # noqa: E402
from src.model import (  # noqa: E402
    MetaLoRA,
    VocabularyHead,
    is_trainable_parameter_name,
)


class _FakeQwen(nn.Module):
    """Minimal stand-in for a Qwen backbone (no transformers needed)."""

    def __init__(self, dim=16, layers=2, vocab=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(
            [nn.ModuleDict({"q_proj": nn.Linear(dim, dim, bias=False)}) for _ in range(layers)]
        )
        self.config = SimpleNamespace(hidden_size=dim, num_hidden_layers=layers)

    def get_input_embeddings(self):
        return self.embed


def _fake_metaloRA(**kwargs):
    torch.manual_seed(0)
    options = dict(
        rank=2,
        alpha=4.0,
        memory_length=2,
        decoder_hidden_size=8,
        decoder_heads=2,
        decoder_ffn_ratio=2,
        target_modules=["q_proj"],
        max_context_tokens=8,
        trainable_dtype=torch.float32,
        context_lm=True,
    )
    options.update(kwargs)
    return MetaLoRA(_FakeQwen(), **options)


def _tied_head(D=8, H=16, V=32, init_adapter=None):
    torch.manual_seed(1)
    embedding = nn.Parameter(torch.randn(V, H), requires_grad=False)
    head = VocabularyHead(
        D, V, mode="tied", embedding_getter=lambda: embedding,
        backbone_hidden_size=H, init_adapter=init_adapter,
    )
    return embedding, head


# --- 1. linear mode is the old behaviour -------------------------------------------


def test_linear_mode_is_a_plain_linear():
    torch.manual_seed(0)
    head = VocabularyHead(8, 32, mode="linear")
    hidden = torch.randn(5, 8)
    assert torch.allclose(head(hidden), F.linear(hidden, head.weight), atol=0, rtol=0)
    assert set(dict(head.named_parameters())) == {"weight"}
    assert head.compute_dtype == head.weight.dtype
    assert head.materialized_weight() is head.weight


def test_default_model_exposes_the_old_parameter_name():
    model = _fake_metaloRA()
    assert isinstance(model.context_lm_head, VocabularyHead)
    assert model.context_lm_head.mode == "linear"
    parameters = dict(model.named_parameters())
    assert "context_lm_head.weight" in parameters
    assert parameters["context_lm_head.weight"].shape == (32, 8)
    assert not any(name.startswith("context_lm_head.adapter") for name in parameters)


def test_linear_state_dict_matches_a_fresh_nn_linear_layout():
    """A default run must stay checkpoint-compatible with the pre-A1 layout."""
    model = _fake_metaloRA()
    reference = nn.Linear(8, 32, bias=False)
    assert model.context_lm_head.weight.shape == reference.weight.shape
    assert model.context_lm_head.weight.dtype == torch.float32


# --- 2. tied mode scores with the frozen embedding ---------------------------------


def test_tied_weight_is_embedding_times_adapter():
    embedding, head = _tied_head()
    expected = embedding @ head.adapter.weight
    assert torch.allclose(head.materialized_weight(), expected, atol=1e-6)
    hidden = torch.randn(3, 8)
    assert torch.allclose(head(hidden), F.linear(hidden, expected), atol=1e-6)
    assert set(dict(head.named_parameters())) == {"adapter.weight"}
    assert head.compute_dtype == head.adapter.weight.dtype


def test_tied_mode_trains_only_the_adapter():
    embedding, head = _tied_head()
    hidden = torch.randn(4, 8, requires_grad=True)
    head(hidden).sum().backward()
    assert head.adapter.weight.grad is not None
    assert head.adapter.weight.grad.abs().sum() > 0
    assert embedding.grad is None


def test_tied_head_materialises_in_its_own_dtype_under_autocast():
    """Regression: under bf16 autocast a float32 matmul comes back bfloat16.

    ``W = E @ adapter`` must stay float32 (the trainable dtype) even though the whole
    training forward runs inside accelerate's autocast; otherwise the fp32 chunk in the
    loss meets a bf16 weight matrix and raises a dtype error.
    """
    embedding, head = _tied_head()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        weight = head.materialized_weight()
        logits = head(torch.randn(3, 8))
    assert weight.dtype == torch.float32, f"materialised W came back {weight.dtype}"
    assert logits.dtype == torch.float32, f"logits came back {logits.dtype}"
    assert torch.allclose(weight, embedding @ head.adapter.weight, atol=1e-6)


def test_tied_head_mixes_backbone_and_trainable_dtypes():
    """The real case: bfloat16 embedding times a float32 adapter."""
    torch.manual_seed(3)
    embedding = nn.Parameter(torch.randn(32, 16).to(torch.bfloat16), requires_grad=False)
    head = VocabularyHead(
        8, 32, mode="tied", embedding_getter=lambda: embedding, backbone_hidden_size=16,
    )
    assert head.compute_dtype == torch.float32
    weight = head.materialized_weight()
    assert weight.dtype == torch.float32
    assert weight.shape == (32, 8)
    assert torch.allclose(weight, embedding.float() @ head.adapter.weight, atol=1e-6)
    hidden = torch.randn(3, 8)                     # float32, as the chunked loss casts
    assert head(hidden).dtype == torch.float32


def test_tied_head_parameter_count_is_adapter_only():
    embedding, head = _tied_head(D=8, H=16, V=32)
    total = sum(p.numel() for p in head.parameters())
    assert total == 16 * 8  # backbone_hidden_size * hidden_size


# --- 3. the materialised weight cannot go stale ------------------------------------


def test_materialised_weight_follows_inplace_updates():
    embedding, head = _tied_head()
    before = head.materialized_weight().clone()
    with torch.no_grad():
        head.adapter.weight.add_(0.25)
    after = head.materialized_weight()
    assert not torch.allclose(before, after)
    assert torch.allclose(after, embedding @ head.adapter.weight, atol=1e-6)


def test_refresh_rebuilds_the_cache():
    embedding, head = _tied_head()
    first = head.materialized_weight()
    assert head.materialized_weight() is first          # cached
    head.refresh()
    second = head.materialized_weight()
    assert second is not first
    assert torch.allclose(second, first, atol=0, rtol=0)


def test_optimizer_step_invalidates_the_cache():
    embedding, head = _tied_head()
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    hidden = torch.randn(6, 8)
    logits_before = head(hidden).detach().clone()
    head(hidden).sum().backward()
    optimizer.step()                                    # in-place update
    logits_after = head(hidden).detach()
    assert not torch.allclose(logits_before, logits_after)


def test_a_no_grad_materialisation_does_not_poison_the_next_training_step():
    """An evaluation must not leave a grad-free weight behind for training to reuse.

    Every evaluation runs inside ``torch.no_grad()``, and the optimizer step just before
    it bumped the adapter's version, so the evaluation always materialises a fresh weight
    -- without a graph. Caching that tensor under the current version made the *next*
    training forward score with a weight the adapter is not part of: the adapter received
    no gradient on that step and DDP aborted the run with

        RuntimeError: Expected to have finished reduction in the prior iteration ...
        Parameter indices which did not receive grad for rank 3: 757

    757 is ``context_lm_head.adapter.weight`` in the 1.7B ON configuration. This test is
    the single-process version of that failure: materialise under ``no_grad`` and check
    that the very next grad-enabled forward still reaches the adapter.
    """
    _, head = _tied_head()
    with torch.no_grad():
        stale = head.materialized_weight()
    assert not stale.requires_grad, "the no_grad product is only useful for the eval itself"

    hidden = torch.randn(6, 8)
    head(hidden).sum().backward()
    assert head.adapter.weight.grad is not None, "the adapter got no gradient after an evaluation"
    assert float(head.adapter.weight.grad.abs().sum()) > 0.0


def test_a_grad_enabled_materialisation_is_still_cached():
    """The perf property must survive the fix: one materialisation per step, not per call."""
    _, head = _tied_head()
    first = head.materialized_weight()
    assert first.requires_grad
    assert head.materialized_weight() is first


# --- 4. memory-projection initialisation -------------------------------------------


def test_memory_projection_init_is_the_up_projection():
    model = _fake_metaloRA(head_mode="tied", head_init="memory_projection")
    head = model.context_lm_head
    embedding = model.qwen.get_input_embeddings().weight
    stacked = torch.stack(
        [decoder.memory_projection.weight for decoder in model.decoders], dim=0
    ).mean(dim=0)
    assert torch.allclose(head.adapter.weight, stacked.t(), atol=1e-6)
    assert torch.allclose(head.materialized_weight(), embedding @ stacked.t(), atol=1e-6)


def test_auto_init_resolves_to_memory_projection_only_when_tied():
    tied = _fake_metaloRA(head_mode="tied", head_init="auto")
    stacked = torch.stack(
        [decoder.memory_projection.weight for decoder in tied.decoders], dim=0
    ).mean(dim=0)
    assert torch.allclose(tied.context_lm_head.adapter.weight, stacked.t(), atol=1e-6)
    linear = _fake_metaloRA(head_mode="linear", head_init="auto")
    assert linear.context_lm_head.mode == "linear"


def test_random_init_is_not_the_memory_projection():
    model = _fake_metaloRA(head_mode="tied", head_init="random")
    stacked = torch.stack(
        [decoder.memory_projection.weight for decoder in model.decoders], dim=0
    ).mean(dim=0)
    assert not torch.allclose(model.context_lm_head.adapter.weight, stacked.t())


def test_tied_head_is_trainable_by_name():
    model = _fake_metaloRA(head_mode="tied")
    names = [name for name, _ in model.named_parameters()]
    assert "context_lm_head.adapter.weight" in names
    assert is_trainable_parameter_name("context_lm_head.adapter.weight")
    model.set_trainable_dtype(torch.float32)
    assert model.context_lm_head.adapter.weight.dtype == torch.float32


# --- 5. the loss path --------------------------------------------------------------


def test_context_lm_loss_runs_and_backpropagates_through_a_tied_head():
    embedding, head = _tied_head()
    hidden = torch.randn(2, 2, 4, 8, requires_grad=True)   # [B, layers, P, D]
    labels = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]])
    loss = context_lm_loss(hidden, labels, head, None, max_logits_rows=2)
    assert torch.isfinite(loss)
    loss.backward()
    assert head.adapter.weight.grad is not None
    assert head.adapter.weight.grad.abs().sum() > 0
    assert embedding.grad is None


def _expect_value_error(function, message):
    try:
        function()
    except ValueError:
        return
    raise AssertionError(message)


def test_tied_head_rejects_bad_arguments():
    _expect_value_error(
        lambda: VocabularyHead(8, 32, mode="bogus"),
        "an unknown mode must raise",
    )
    _expect_value_error(
        lambda: VocabularyHead(8, 32, mode="tied", backbone_hidden_size=16),
        "tied mode without an embedding getter must raise",
    )
    embedding, _ = _tied_head()
    _expect_value_error(
        lambda: VocabularyHead(
            8, 32, mode="tied", embedding_getter=lambda: embedding,
            backbone_hidden_size=16, init_adapter=torch.zeros(3, 3),
        ),
        "a mismatched init_adapter shape must raise",
    )


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"ok  {name}")
    print("all vocab-head tests passed")
