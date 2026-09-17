"""Memory-prefixed autoencoding objective (B2).

Run with either ``python tests/test_ae_lm.py`` or ``pytest tests``.

What is asserted
----------------
1. ``sequence_lm_loss`` scores ``hidden[i]`` against ``labels[i]`` directly (the caller
   shifts once), ignores ``-100``, honours an extra mask, agrees with a manual
   cross entropy, and is invariant to the chunk size.
2. ``positions`` restricts the loss to the gathered targets and returns exactly the same
   number as masking every other position -- including for a row with fewer valid positions
   than the budget, whose unused slots ``sample_positions`` marks invalid.
3. Gradient reaches the tied unembedding's adapter (or a plain linear head) and the hidden
   states, through the checkpointed chunk path.
4. On a real (tiny) Qwen3: ``ae_lm=True`` populates aligned autoencoding terms, the loss is
   finite, ``memory_tokens`` receive gradient while the frozen backbone does not, and the
   predictions actually depend on the memory prefix (changing the slots changes the loss).
5. ``ae_lm=False`` (the default) computes none of it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.losses import sample_positions, sequence_lm_loss  # noqa: E402
from src.model import MetaLoRA, TiedUnembedding, VocabularyHead, is_trainable_parameter_name  # noqa: E402

try:
    from transformers import Qwen3Config, Qwen3ForCausalLM
except ImportError:  # pragma: no cover - transformers is optional for the unit suite
    Qwen3Config = Qwen3ForCausalLM = None


def _head(D=8, V=32, H=16, tied=True):
    torch.manual_seed(0)
    embedding = nn.Parameter(torch.randn(V, H), requires_grad=False)
    if tied:
        return embedding, VocabularyHead(
            D, V, mode="tied", embedding_getter=lambda: embedding,
            backbone_hidden_size=H, init_adapter=torch.randn(H, D),
        )
    return embedding, VocabularyHead(D, V, mode="linear")


def _manual_ce(hidden, labels, weight):
    """Reference cross entropy. Flattens first: F.cross_entropy reads dim 1 as the class
    dimension for a 3-D input, which is why the real losses reshape to ``[rows, vocab]``."""
    logits = F.linear(hidden, weight).float()
    flat = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
        ignore_index=-100, reduction="none",
    )
    active = labels.reshape(-1).ne(-100)
    return flat[active].sum() / active.sum().clamp_min(1)


# --- 1. loss semantics -------------------------------------------------------------


def test_loss_matches_manual_cross_entropy_without_a_shift():
    embedding, head = _head()
    torch.manual_seed(1)
    hidden = torch.randn(2, 6, 8)
    labels = torch.randint(0, 32, (2, 6))
    labels[0, 2] = -100                       # ignored
    got = sequence_lm_loss(hidden, labels, head, max_logits_rows=4)
    want = _manual_ce(hidden, labels, head.materialized_weight())
    assert torch.allclose(got, want, atol=1e-6), f"{float(got)} vs {float(want)}"


def test_extra_mask_restricts_the_targets():
    embedding, head = _head()
    torch.manual_seed(2)
    hidden = torch.randn(1, 5, 8)
    labels = torch.randint(0, 32, (1, 5))
    mask = torch.tensor([[True, False, True, False, True]])
    got = sequence_lm_loss(hidden, labels, head, mask=mask, max_logits_rows=8)
    want = _manual_ce(hidden[:, mask[0]], labels[:, mask[0]], head.materialized_weight())
    assert torch.allclose(got, want, atol=1e-6)


def test_chunk_size_does_not_change_the_loss():
    embedding, head = _head()
    torch.manual_seed(3)
    hidden = torch.randn(2, 7, 8)
    labels = torch.randint(0, 32, (2, 7))
    small = sequence_lm_loss(hidden, labels, head, max_logits_rows=1)
    large = sequence_lm_loss(hidden, labels, head, max_logits_rows=64)
    assert torch.allclose(small, large, atol=1e-5)


def test_positions_select_the_same_targets_as_a_mask():
    embedding, head = _head()
    torch.manual_seed(4)
    hidden = torch.randn(1, 6, 8)
    labels = torch.randint(0, 32, (1, 6))
    positions = torch.tensor([[1, 4]])
    got = sequence_lm_loss(hidden, labels, head, positions=positions)
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask.scatter_(1, positions, True)
    want = sequence_lm_loss(hidden, labels, head, mask=mask)
    assert torch.allclose(got, want, atol=1e-6)


def test_a_row_with_fewer_positions_than_the_budget_scores_each_one_once():
    """The unused sample slots of a short row must not be scored.

    ``sample_positions`` fills them with index 0 and marks them False. The keep-mask has to
    be gathered alongside the hidden states: before it was, those duplicates pointed at a
    valid label and were counted, so a context shorter than ``ae_lm_positions`` was trained
    mostly on its own first token (3 valid positions under a budget of 5 put token 0 into 3
    of the 5 loss terms).
    """
    embedding, head = _head()
    torch.manual_seed(6)
    hidden = torch.randn(1, 5, 8)
    labels = torch.tensor([[7, 11, 13, -100, -100]])
    mask = labels.ne(-100)
    positions, keep = sample_positions(mask, 5)
    assert int(keep.sum()) == 3, "the sampler marks the two duplicate slots invalid"
    assert int((positions < 0).sum()) == 2, "unused slots carry -1, not a repeated index 0"

    sampled = sequence_lm_loss(hidden, labels, head, mask=mask, positions=positions)
    dense = sequence_lm_loss(hidden, labels, head, mask=mask)
    assert torch.allclose(sampled, dense, atol=1e-6), "the duplicate slots changed the loss"

    per_token = F.cross_entropy(
        F.linear(hidden[0, :3], head.materialized_weight()), labels[0, :3], reduction="none"
    )
    assert torch.allclose(sampled, per_token.mean(), atol=1e-6)


def test_loss_is_finite_and_gradients_flow_through_the_checkpoint():
    embedding, head = _head()
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    labels = torch.randint(0, 32, (2, 5))
    loss = sequence_lm_loss(hidden, labels, head, max_logits_rows=2)
    assert torch.isfinite(loss)
    loss.backward()
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0
    assert head.adapter.weight.grad is not None and head.adapter.weight.grad.abs().sum() > 0
    assert embedding.grad is None


def test_empty_targets_give_zero_not_nan():
    embedding, head = _head()
    hidden = torch.randn(1, 4, 8, requires_grad=True)
    labels = torch.full((1, 4), -100)
    loss = sequence_lm_loss(hidden, labels, head)
    assert torch.isfinite(loss)
    loss.backward()
    assert hidden.grad is not None


def test_sample_positions_respects_the_mask():
    torch.manual_seed(5)
    mask = torch.zeros(3, 10, dtype=torch.bool)
    mask[0, :4] = True
    mask[1, 5:7] = True
    mask[2, 9] = True
    positions, keep = sample_positions(mask, 3)
    assert positions.shape == (3, 3) and keep.shape == (3, 3)
    for row in range(3):
        chosen = keep[row]
        picked = positions[row][chosen]
        assert bool(mask[row][picked].all()), "picked a masked-out slot"
        assert int(chosen.sum()) == min(3, int(mask[row].sum()))
        # Slots the row could not fill are -1, not a repeated index 0 (see the sampler's
        # docstring: a repeated 0 would be scored again by a caller that forgot the mask).
        assert bool((positions[row][~chosen] < 0).all())


# --- 2. the tied unembedding -------------------------------------------------------


def test_tied_unembedding_has_no_parameters():
    embedding = nn.Parameter(torch.randn(32, 16), requires_grad=False)
    head = TiedUnembedding(lambda: embedding)
    assert list(head.parameters()) == []
    torch.manual_seed(6)
    hidden = torch.randn(3, embedding.size(1))          # last dim is the backbone width
    assert head.compute_dtype == embedding.dtype
    assert torch.allclose(head(hidden), F.linear(hidden, embedding), atol=0, rtol=0)
    assert torch.allclose(head.materialized_weight(), embedding, atol=0, rtol=0)


# --- 3. the autoencoding pass on a real backbone -----------------------------------


def _tiny_qwen3(vocab=64, hidden=32, layers=2, heads=4, kv_heads=2, intermediate=64):
    config = Qwen3Config(
        vocab_size=vocab,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        intermediate_size=intermediate,
        max_position_embeddings=256,
        tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(config)


def _model(ae_lm=True, memory_length=4):
    torch.manual_seed(0)
    model = MetaLoRA(
        _tiny_qwen3(),
        rank=2,
        alpha=4.0,
        memory_length=memory_length,
        decoder_hidden_size=16,
        decoder_heads=4,
        decoder_ffn_ratio=2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        dropout=0.0,
        max_context_tokens=16,
        trainable_dtype=torch.float32,
        context_lm=True,
        ae_lm=ae_lm,
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
        embedding(batch["context_ids"]),
        batch["context_mask"],
        embedding(batch["question_ids"]),
        batch["question_mask"],
        embedding(batch["answer_ids"]),
        batch["answer_mask"],
        batch["answer_ids"],
        torch.arange(batch["context_ids"].size(0)),
        context_ids=batch["context_ids"],
        context_lm_positions=4,
    )


def test_autoencoding_terms_are_aligned_and_depend_on_the_memory():
    if Qwen3ForCausalLM is None:
        print("skip: transformers is not installed")
        return
    model = _model(ae_lm=True)
    batch = _batch()
    output = _forward(model, batch)
    L = batch["context_ids"].size(1)
    assert output.ae_hidden.shape == (2, L - 1, 32)
    assert torch.equal(output.ae_labels, batch["context_ids"][:, 1:])
    assert torch.equal(output.ae_mask, batch["context_mask"][:, 1:])

    loss_before = sequence_lm_loss(
        output.ae_hidden, output.ae_labels, model.ae_head, output.ae_mask, max_logits_rows=8
    )
    assert torch.isfinite(loss_before)

    # The predictions must be conditioned on the memory prefix: change the slots and the
    # loss must move.
    with torch.no_grad():
        model.memory_tokens.add_(0.1)
    other = _forward(model, batch)
    loss_after = sequence_lm_loss(
        other.ae_hidden, other.ae_labels, model.ae_head, other.ae_mask, max_logits_rows=8
    )
    assert not torch.allclose(loss_before, loss_after), "the memory prefix is not being used"


def test_autoencoding_gradient_reaches_the_memory_only():
    if Qwen3ForCausalLM is None:
        print("skip: transformers is not installed")
        return
    model = _model(ae_lm=True)
    batch = _batch()
    output = _forward(model, batch)
    loss = sequence_lm_loss(
        output.ae_hidden, output.ae_labels, model.ae_head, output.ae_mask, max_logits_rows=8
    )
    loss.backward()
    assert model.memory_tokens.grad is not None and model.memory_tokens.grad.abs().sum() > 0
    assert model.ae_head is not None and list(model.ae_head.parameters()) == []
    assert all(p.grad is None for p in model.qwen.parameters() if not p.requires_grad)


def test_ae_lm_off_computes_nothing():
    if Qwen3ForCausalLM is None:
        print("skip: transformers is not installed")
        return
    model = _model(ae_lm=False)
    assert model.ae_head is None
    output = _forward(model, _batch())
    assert output.ae_hidden is None and output.ae_labels is None and output.ae_mask is None


def test_autoencoding_needs_context_ids():
    if Qwen3ForCausalLM is None:
        print("skip: transformers is not installed")
        return
    model = _model(ae_lm=True)
    batch = _batch()
    embedding = model.qwen.get_input_embeddings()
    try:
        model(
            embedding(batch["context_ids"]), batch["context_mask"],
            embedding(batch["question_ids"]), batch["question_mask"],
            embedding(batch["answer_ids"]), batch["answer_mask"],
            batch["answer_ids"],
        )
    except ValueError:
        return
    raise AssertionError("ae_lm without context_ids must raise")


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"ok  {name}")
    print("all autoencoding tests passed")
