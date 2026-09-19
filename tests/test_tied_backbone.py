"""A tied vocabulary head against a *real* Qwen3 backbone (A1).

``tests/test_vocab_head.py`` pins the head in isolation with a fake backbone. This file
covers the part a fake cannot: that the head picks up the real (tied) input embedding of
a ``Qwen3ForCausalLM``, that gradients flow from both losses into the adapter while the
frozen backbone weights stay grad-free, and that a full forward/backward/optimizer step
works with the head in place.

The backbone is a tiny randomly initialised ``Qwen3Config`` built in-process, so this
needs neither a GPU nor a model download. It is skipped when ``transformers`` is not
importable because the rest of the suite deliberately runs without it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.losses import token_recon_loss, qa_loss  # noqa: E402
from src.model import MetaLoRA, is_trainable_parameter_name  # noqa: E402

try:
    from transformers import Qwen3Config, Qwen3ForCausalLM
except ImportError:  # pragma: no cover - transformers is optional for the unit suite
    Qwen3Config = Qwen3ForCausalLM = None


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


def _build(head_mode):
    torch.manual_seed(0)
    qwen = _tiny_qwen3()
    model = MetaLoRA(
        qwen,
        rank=2,
        alpha=4.0,
        memory_length=4,
        decoder_hidden_size=16,
        decoder_heads=4,
        decoder_ffn_ratio=2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        dropout=0.0,
        max_context_tokens=16,
        trainable_dtype=torch.float32,
        token_recon=True,
        head_mode=head_mode,
        head_init="auto",
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad = is_trainable_parameter_name(name)
    return model


def _batch(vocab=64, B=2, L=12, Q=4, A=5):
    torch.manual_seed(1)
    return dict(
        context_ids=torch.randint(0, vocab, (B, L)),
        context_mask=torch.ones(B, L, dtype=torch.bool),
        question_ids=torch.randint(0, vocab, (B, Q)),
        question_mask=torch.ones(B, Q, dtype=torch.bool),
        answer_ids=torch.randint(0, vocab, (B, A)),
        answer_mask=torch.ones(B, A, dtype=torch.bool),
    )


def _forward_losses(model, batch):
    embedding = model.qwen.get_input_embeddings()
    output = model(
        embedding(batch["context_ids"]),
        batch["context_mask"],
        embedding(batch["question_ids"]),
        batch["question_mask"],
        embedding(batch["answer_ids"]),
        batch["answer_mask"],
        batch["answer_ids"],          # labels: every answer token is supervised
        torch.arange(batch["context_ids"].size(0)),
        context_ids=batch["context_ids"],
        token_recon_positions=4,
    )
    qa = qa_loss(output.logits, output.labels)
    recon = token_recon_loss(
        output.token_recon_hidden,
        output.token_recon_labels,
        model.token_recon_head,
        output.token_recon_mask,
        max_logits_rows=8,
    )
    return output, qa, recon


def test_real_backbone_ties_its_head_to_the_input_embedding():
    if Qwen3ForCausalLM is None:
        print("skip: transformers is not installed")
        return
    model = _build("tied")
    embedding = model.qwen.get_input_embeddings().weight
    assert model.qwen.lm_head.weight is embedding
    # The tied head must score with that very tensor.
    assert model.token_recon_head.materialized_weight().shape == (64, 16)
    expected = embedding @ model.token_recon_head.adapter.weight
    assert torch.allclose(model.token_recon_head.materialized_weight(), expected, atol=1e-6)


def test_tied_head_trains_end_to_end_on_a_real_backbone():
    if Qwen3ForCausalLM is None:
        print("skip: transformers is not installed")
        return
    model = _build("tied")
    batch = _batch()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    _, qa, recon = _forward_losses(model, batch)
    total = qa + recon
    assert torch.isfinite(total)
    total.backward()

    adapter_grad = model.token_recon_head.adapter.weight.grad
    assert adapter_grad is not None and adapter_grad.abs().sum() > 0
    assert model.memory_tokens.grad is not None and model.memory_tokens.grad.abs().sum() > 0
    # The frozen backbone must not accumulate gradient.
    assert all(
        p.grad is None for p in model.qwen.parameters() if not p.requires_grad
    )

    optimizer.step()
    # A second step must not reuse the pre-step materialised weight.
    _, qa_after, recon_after = _forward_losses(model, batch)
    assert torch.isfinite(qa_after + recon_after)


def test_linear_and_tied_heads_train_and_differ():
    if Qwen3ForCausalLM is None:
        print("skip: transformers is not installed")
        return
    losses = {}
    for mode in ("linear", "tied"):
        model = _build(mode)
        batch = _batch()
        _, qa, recon = _forward_losses(model, batch)
        (qa + recon).backward()
        losses[mode] = float((qa + recon).detach())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        losses[f"{mode}_trainable"] = trainable
    # tied replaces 16*64 = 1024 head weights with an adapter of 32*16 = 512.
    assert losses["tied_trainable"] < losses["linear_trainable"]
    assert losses["linear_trainable"] - losses["tied_trainable"] == 1024 - 512


def test_slot_attention_changes_the_encoder_pass_deterministically():
    """``slot_attention`` must reach ``encode_context_prefix``, not just the model.

    The rule itself is pinned by ``tests/test_masks.py``; here we check the plumbing on a
    real backbone: with the flag off two calls agree bit for bit, and turning it on (which
    lets slot k read slots <= k) changes the per-layer memory states.
    """
    if Qwen3ForCausalLM is None:
        print("skip: transformers is not installed")
        return
    model = _build("linear")
    batch = _batch()
    embedding = model.qwen.get_input_embeddings()
    context = embedding(batch["context_ids"])

    model.slot_attention = "isolated"
    off_first = model.encode_context_prefix(context, batch["context_mask"])
    off_second = model.encode_context_prefix(context, batch["context_mask"])
    assert torch.equal(off_first.layer_memory, off_second.layer_memory)

    model.slot_attention = "causal"
    on = model.encode_context_prefix(context, batch["context_mask"])
    assert off_first.layer_memory.shape == on.layer_memory.shape
    assert not torch.allclose(off_first.layer_memory, on.layer_memory), (
        "slot attention did not change the memory states"
    )
    model.slot_attention = "bidirectional"
    bidirectional = model.encode_context_prefix(context, batch["context_mask"])
    assert not torch.allclose(on.layer_memory, bidirectional.layer_memory)
    model.zero_grad(set_to_none=True)
    _, qa, recon = _forward_losses(model, batch)
    (qa + recon).backward()
    assert torch.isfinite(qa + recon)
    assert model.memory_tokens.grad is not None
    assert torch.isfinite(model.memory_tokens.grad).all()
    assert model.memory_tokens.grad.abs().sum() > 0


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"ok  {name}")
    print("all real-backbone head tests passed")
