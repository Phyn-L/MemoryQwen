"""Encoder -> learned-query writer -> short Qwen KV contracts."""
import copy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.MemoryDecoder import MemoryDecoder
from src.model import MetaLoRA, load_model
from utils.config import TrainConfig


def tiny_model(**kwargs):
    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=64, tie_word_embeddings=True,
    )
    config._attn_implementation = "sdpa"
    options = dict(
        memory_length=4, target_modules=["q_proj", "v_proj"], rank=2,
        memory_resampler_hidden_size=16,
        memory_resampler_heads=2, readout_hidden_size=16, readout_heads=2,
    )
    options.update(kwargs)
    return MetaLoRA(Qwen3ForCausalLM(config), **options)


def encode(model, length=17, batch=2):
    ids = torch.randint(0, 64, (batch, length))
    return model.encode_context_prefix(
        model.qwen.get_input_embeddings()(ids), torch.ones_like(ids, dtype=torch.bool),
    )


def test_exactly_one_context_and_one_short_memory_forward():
    model = tiny_model().eval()
    calls = []
    hook = model._transformer_body.register_forward_pre_hook(
        lambda _m, _args, kwargs: calls.append(kwargs.copy()), with_kwargs=True,
    )
    prefix = encode(model, length=64)
    hook.remove()
    assert [c["inputs_embeds"].shape[1] for c in calls] == [64, 4]
    assert calls[0]["use_cache"] is False
    assert calls[0]["output_hidden_states"] is False
    assert calls[0]["attention_mask"] is None
    assert prefix.memory.shape == prefix.memory_embedding.shape == (2, 4, 32)
    assert prefix.layer_memory.shape == (2, 3, 4, 32)
    assert prefix.context_hidden is None
    assert not hasattr(prefix, "context_target")
    assert all(layer.keys.shape[-2] == layer.values.shape[-2] == 4
               for layer in prefix.memory_cache.layers)


def test_layer_mixture_matches_explicit_hidden_states():
    model = tiny_model(context_layer_indices=(-1, 0)).eval()
    model.context_layer_logits.data.copy_(torch.tensor([0.4, -0.6]))
    embeds = model.qwen.get_input_embeddings()(torch.randint(0, 64, (2, 17)))
    mask = torch.ones(2, 17, dtype=torch.bool)
    positions = torch.arange(17)[None].expand(2, -1)
    expected = model._transformer_body(
        inputs_embeds=embeds, position_ids=positions, output_hidden_states=True, use_cache=False,
    ).hidden_states
    hooks_before = [set(layer._forward_hooks) for layer in model._transformer_body.layers]
    actual, final = model._encode_context_hidden(embeds, mask, positions)
    weights = model.context_layer_logits.softmax(0)
    torch.testing.assert_close(actual, weights[0] * expected[-1] + weights[1] * expected[1])
    torch.testing.assert_close(final, expected[-1])
    actual.square().mean().backward()
    assert model.context_layer_logits.grad.abs().sum() > 0
    assert [set(layer._forward_hooks) for layer in model._transformer_body.layers] == hooks_before


def test_padding_content_cannot_change_memory():
    model = tiny_model().eval()
    embeds = torch.randn(2, 17, 32)
    mask = torch.ones(2, 17, dtype=torch.bool)
    mask[0, 8:] = False
    mask[1, :4] = False
    before = model.encode_context_prefix(embeds, mask)
    altered = embeds.clone()
    altered[~mask] = torch.randn_like(altered[~mask]) * 100
    after = model.encode_context_prefix(altered, mask)
    torch.testing.assert_close(before.memory, after.memory)


@pytest.mark.parametrize("source", ["memory", "memory_embedding", "layer_mean"])
def test_qa_gradients_reach_writer_lora_layer_weights_and_reader(source):
    model = tiny_model(
        context_layer_indices=(-1, -2), reader_memory_source=source, readout_length=2,
    )
    prefix = encode(model)
    question = model.qwen.get_input_embeddings()(torch.randint(0, 64, (2, 3)))
    answer = model.qwen.get_input_embeddings()(torch.randint(0, 64, (2, 2)))
    output = model.forward_qa_with_prefix(
        prefix, torch.arange(2), question, torch.ones(2, 3, dtype=torch.bool),
        answer, torch.ones(2, 2, dtype=torch.bool), torch.ones(2, 2, dtype=torch.long),
    )
    output.logits.square().mean().backward()
    for parameter in (
        model.context_resampler.latents, model.context_layer_logits, model.resampler.latents,
    ):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for name, p in model.named_parameters() if "lora_B" in name)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)


def test_context_checkpointing_preserves_cache_outputs_and_gradients():
    plain = tiny_model(context_layer_indices=(-1, 0)).train()
    checked = copy.deepcopy(plain)
    checked.context_gradient_checkpointing = True
    embeds = torch.randn(2, 17, 32)
    mask = torch.ones(2, 17, dtype=torch.bool)
    for model in (plain, checked):
        prefix = model.encode_context_prefix(embeds, mask)
        prefix.memory.square().mean().backward()
        assert prefix.memory_cache.get_seq_length() == 4
        assert not any(layer.gradient_checkpointing for layer in model._transformer_body.layers)
    for (name, p), (_, q) in zip(plain.named_parameters(), checked.named_parameters()):
        if p.grad is not None:
            assert q.grad is not None, name
            torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4)


def test_context_checkpointing_restores_state_after_forward_failure(monkeypatch):
    model = tiny_model(context_layer_indices=(-1, 0), context_gradient_checkpointing=True).train()
    layers = model._transformer_body.layers
    before = [(layer.gradient_checkpointing, getattr(layer, "_gradient_checkpointing_func", None),
               set(layer._forward_hooks)) for layer in layers]

    def fail(*args, **kwargs):
        raise RuntimeError("injected forward failure")

    monkeypatch.setattr(model._transformer_body, "forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        encode(model)
    after = [(layer.gradient_checkpointing, getattr(layer, "_gradient_checkpointing_func", None),
              set(layer._forward_hooks)) for layer in layers]
    assert after == before


def test_peft_writer_forward_backward():
    pytest.importorskip("peft")
    model = tiny_model(use_peft=True, context_gradient_checkpointing=True).train()
    prefix = encode(model)
    prefix.memory.square().mean().backward()
    assert model.context_resampler.latents.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for name, p in model.named_parameters() if "lora_B" in name)


def test_invalid_config_is_rejected_before_loading_tokenizer(monkeypatch):
    from transformers import AutoTokenizer

    def fail(*args, **kwargs):
        pytest.fail("tokenizer must not load for an invalid configuration")

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", fail)
    cfg = TrainConfig()
    cfg.memory.memory_resampler_layers = 0
    with pytest.raises(ValueError, match="resampler"):
        load_model(cfg)


def test_ae_config_window_is_validated_during_loading(monkeypatch):
    from types import SimpleNamespace
    from transformers import AutoModelForCausalLM, AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained",
                        lambda *a, **kw: SimpleNamespace(pad_token=None, eos_token="<eos>"))
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained",
                        lambda *a, **kw: SimpleNamespace(config=SimpleNamespace(max_position_embeddings=64)))
    cfg = TrainConfig()
    cfg.data.max_context_tokens = 64
    cfg.memory.causal_recon_weight = 1.0
    with pytest.raises(ValueError, match="backbone window"):
        load_model(cfg)


@pytest.mark.parametrize("indices", [(), (3,), (-4,), (-1, 2), (0.5,), (True,)])
def test_invalid_layer_selection_fails_early(indices):
    with pytest.raises(ValueError, match="context_layer_indices"):
        tiny_model(context_layer_indices=indices)


def test_window_and_empty_context_validation():
    model = tiny_model()
    with pytest.raises(ValueError, match="window"):
        encode(model, length=65)
    with pytest.raises(ValueError, match="at least one"):
        model.encode_context_prefix(torch.randn(2, 4, 32), torch.zeros(2, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match="context_mask"):
        model.encode_context_prefix(torch.randn(2, 4, 32), torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="window"):
        tiny_model(max_context_tokens=65)


def test_reconstruction_queries_have_no_learned_position_capacity():
    decoder = MemoryDecoder(32, 16, 2)
    output = decoder.decode(torch.randn(2, 4, 32), torch.tensor([0, 2048, 40959]))
    assert output.shape == (2, 3, 16) and torch.isfinite(output).all()
    assert not torch.equal(output[:, 1], output[:, 2])


def test_writer_config_roundtrip():
    config = TrainConfig.from_dict({"memory": {
        "context_layer_indices": [-1, -3], "reader_memory_source": "memory_embedding",
    }, "data": {"max_context_tokens": 40960}})
    config.validate()
    assert TrainConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()
    for field, value in (("memory_resampler_hidden_size", 0), ("memory_resampler_ffn_ratio", 0),
                         ("memory_resampler_heads", 0), ("context_layer_indices", [0.5])):
        config = TrainConfig.from_dict({"memory": {field: value}})
        with pytest.raises(ValueError):
            config.validate()
