"""Encoder -> learned-query writer -> resampled memory embedding contracts."""
import copy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.model import ContextMemoryModel, MetaLoRA, load_model
from utils.config import TrainConfig


def tiny_model(vocab_size=64, num_hidden_layers=3, **kwargs):
    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=num_hidden_layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=64, tie_word_embeddings=True,
    )
    config._attn_implementation = "sdpa"
    options = dict(
        memory_length=4, target_modules=["q_proj", "v_proj"], rank=2,
        memory_resampler_hidden_size=16,
        memory_resampler_heads=2, question_resampler_hidden_size=16,
        question_resampler_heads=2,
        recon_user_token_ids=(2,), recon_answer_prefix_token_ids=(3, 4),
    )
    options.update(kwargs)
    return ContextMemoryModel(Qwen3ForCausalLM(config), **options)


def encode(model, length=17, batch=2):
    ids = torch.randint(0, 64, (batch, length))
    return model.encode_context_prefix(
        model.qwen.get_input_embeddings()(ids), torch.ones_like(ids, dtype=torch.bool),
    )


def test_context_encoding_produces_memory_without_a_separate_backbone_pass():
    model = tiny_model().eval()
    calls = []
    hook = model._transformer_body.register_forward_pre_hook(
        lambda _m, _args, kwargs: calls.append(kwargs.copy()), with_kwargs=True,
    )
    prefix = encode(model, length=64)
    hook.remove()
    assert [c["inputs_embeds"].shape[1] for c in calls] == [64]
    assert calls[0]["use_cache"] is False
    assert calls[0]["output_hidden_states"] is False
    assert calls[0]["attention_mask"] is None
    assert prefix.memory.shape == prefix.memory_embedding.shape == (2, 4, 32)
    assert not hasattr(prefix, "context_hidden")
    assert not hasattr(prefix, "context_target")


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
    torch.testing.assert_close(
        actual, weights[0] * expected[-1] + weights[1] * model._transformer_body.norm(expected[1])
    )
    torch.testing.assert_close(final, expected[-1])
    actual.square().mean().backward()
    assert model.context_layer_logits.grad.abs().sum() > 0
    assert [set(layer._forward_hooks) for layer in model._transformer_body.layers] == hooks_before


def test_single_final_layer_fusion_matches_the_original_final_hidden_state():
    model = tiny_model(context_layer_indices=(-1,)).eval()
    embeds = model.qwen.get_input_embeddings()(torch.randint(0, 64, (2, 17)))
    mask = torch.ones(2, 17, dtype=torch.bool)
    positions = torch.arange(17)[None].expand(2, -1)
    expected = model._transformer_body(
        inputs_embeds=embeds, position_ids=positions, use_cache=False,
        return_dict=True,
    ).last_hidden_state

    actual, final = model._encode_context_hidden(embeds, mask, positions)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(final, expected)


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


@pytest.mark.parametrize("source", ["hidden", "embedding"])
def test_qa_gradients_reach_writer_lora_layer_weights_and_reader(source):
    model = tiny_model(
        context_layer_indices=(-1, -2), question_resampler_info_source=source, readout_length=2,
    )
    prefix = encode(model)
    question = model.qwen.get_input_embeddings()(torch.randint(0, 64, (2, 3)))
    answer = model.qwen.get_input_embeddings()(torch.randint(0, 64, (2, 2)))
    output = model.forward_with_memory_prefix(
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


def test_embedding_reader_resamples_memory_and_question_embeddings_directly():
    model = tiny_model(question_resampler_info_source="embedding", readout_length=2)
    prefix = encode(model)
    question = model.qwen.get_input_embeddings()(torch.randint(0, 64, (1, 3)))
    captured = []
    hook = model.resampler.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )
    model.forward_with_memory_prefix(
        prefix, torch.tensor([0]), question, torch.ones(1, 3, dtype=torch.bool),
        model.qwen.get_input_embeddings()(torch.randint(0, 64, (1, 2))),
        torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 2, dtype=torch.long),
    )
    hook.remove()
    expected = torch.cat([prefix.memory_embedding[:1], question], dim=1)
    torch.testing.assert_close(captured[0], expected)


def test_context_checkpointing_preserves_memory_outputs_and_gradients():
    plain = tiny_model(context_layer_indices=(-1, 0)).train()
    checked = copy.deepcopy(plain)
    checked.context_gradient_checkpointing = True
    embeds = torch.randn(2, 17, 32)
    mask = torch.ones(2, 17, dtype=torch.bool)
    for model in (plain, checked):
        prefix = model.encode_context_prefix(embeds, mask)
        prefix.memory.square().mean().backward()
        assert prefix.memory.shape[1] == 4
        assert not any(layer.gradient_checkpointing for layer in model._transformer_body.layers)
    for (name, p), (_, q) in zip(
        plain.named_parameters(), checked.named_parameters(), strict=True
    ):
        if p.grad is not None:
            assert q.grad is not None, name
            torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("interval", [1, 2, 4, 6])
def test_context_checkpointing_uses_transformers_selected_layers(interval, monkeypatch):
    model = tiny_model(
        num_hidden_layers=7,
        context_gradient_checkpointing=True,
        gradient_checkpoint_every_n_layers=interval,
    ).train()
    observed = []
    body = model._transformer_body
    layers = body.layers
    enable = body.gradient_checkpointing_enable
    options = []

    def record_options(**kwargs):
        options.append(kwargs)
        enable(**kwargs)

    monkeypatch.setattr(body, "gradient_checkpointing_enable", record_options)

    def observe_checkpointed_layers(*_):
        assert body.layers is layers
        observed.append([layer.gradient_checkpointing for layer in layers])

    handle = body.register_forward_pre_hook(observe_checkpointed_layers)
    try:
        encode(model).memory.square().mean().backward()
    finally:
        handle.remove()
    assert observed == [[index % interval == 0 for index in range(7)]]
    assert options == [{
        "gradient_checkpointing_kwargs": {"use_reentrant": False, "preserve_rng_state": True},
        "every_n_layers": interval,
    }]
    assert not any(layer.gradient_checkpointing for layer in layers)
    for _ in range(2):
        encode(model).memory.square().mean().backward()
    embedding = model.qwen.get_input_embeddings()
    assert not embedding._forward_hooks
    assert not embedding(torch.zeros(1, 1, dtype=torch.long)).requires_grad


@pytest.mark.parametrize("interval", [1, 2, 4, 6])
@pytest.mark.parametrize("padded", [False, True])
def test_transformers_checkpointing_matches_loss_and_trainable_gradients(interval, padded):
    plain = tiny_model(
        num_hidden_layers=7, context_layer_indices=(-1, 0, 3),
        readout_length=2, dropout=0.1,
    ).train()
    for name, parameter in plain.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.02)
    checked = copy.deepcopy(plain)
    checked.context_gradient_checkpointing = True
    checked.gradient_checkpoint_every_n_layers = interval
    embeds = torch.randn(2, 17, 32)
    mask = torch.ones(2, 17, dtype=torch.bool)
    if padded:
        mask[0, -3:] = False
    question_ids = torch.randint(0, 64, (2, 3))
    answer_ids = torch.randint(0, 64, (2, 2))
    context_labels = torch.randint(0, 64, (2, 17)).masked_fill(~mask, -100)
    losses = []
    for model in (plain, checked):
        torch.manual_seed(42)
        prefix = model.encode_context_prefix(embeds, mask)
        embedding = model.qwen.get_input_embeddings()
        qa = model.forward_with_memory_prefix(
            prefix, torch.arange(2), embedding(question_ids), torch.ones_like(question_ids).bool(),
            embedding(answer_ids), torch.ones_like(answer_ids).bool(), answer_ids,
            loss_only=True,
        )
        recon = model.forward_recon_with_memory_prefix(
            prefix, torch.arange(2), embeds, mask, context_labels, loss_only=True,
        )
        loss = qa.supervision_losses.mean() + recon.supervision_losses.mean()
        losses.append(loss.detach())
        loss.backward()
    torch.testing.assert_close(losses[0], losses[1], atol=1e-6, rtol=1e-5)
    plain_parameters = dict(plain.named_parameters())
    checked_parameters = dict(checked.named_parameters())
    assert plain_parameters.keys() == checked_parameters.keys()
    for name, parameter in plain_parameters.items():
        if not parameter.requires_grad:
            continue
        other = checked_parameters[name]
        assert (parameter.grad is None) == (other.grad is None), name
        assert parameter.grad is not None, name
        if parameter.grad is not None:
            torch.testing.assert_close(
                parameter.grad, other.grad, atol=1e-6, rtol=1e-4, msg=name,
            )


def test_context_checkpointing_supports_mixed_attention_types():
    model = tiny_model(context_gradient_checkpointing=True).train()
    model.qwen.config.layer_types = [
        "full_attention",
        "sliding_attention",
        "full_attention",
    ]
    model.qwen.config.sliding_window = 4
    model._transformer_body.has_sliding_layers = True
    model._transformer_body.layers[1].self_attn.sliding_window = 4
    model.gradient_checkpoint_every_n_layers = 2
    plain = copy.deepcopy(model)
    plain.context_gradient_checkpointing = False
    embeds = torch.randn(2, 17, 32)
    mask = torch.ones(2, 17, dtype=torch.bool)
    outputs = []
    for candidate in (plain, model):
        prefix = candidate.encode_context_prefix(embeds, mask)
        outputs.append(prefix.memory)
        prefix.memory.square().mean().backward()
    torch.testing.assert_close(outputs[0], outputs[1])
    for (name, parameter), (_, other) in zip(
        plain.named_parameters(), model.named_parameters(), strict=True,
    ):
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, atol=1e-6, rtol=1e-4, msg=name)


def test_context_checkpointing_restores_state_after_forward_failure(monkeypatch):
    model = tiny_model(context_layer_indices=(-1, 0), context_gradient_checkpointing=True).train()
    layers = model._transformer_body.layers
    before = [(layer.gradient_checkpointing, set(layer._forward_hooks)) for layer in layers]
    embedding_hooks = set(model.qwen.get_input_embeddings()._forward_hooks)

    def fail(*args, **kwargs):
        raise RuntimeError("injected forward failure")

    monkeypatch.setattr(model._transformer_body, "forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        encode(model)
    after = [(layer.gradient_checkpointing, set(layer._forward_hooks)) for layer in layers]
    assert after == before
    assert set(model.qwen.get_input_embeddings()._forward_hooks) == embedding_hooks


@pytest.mark.parametrize("interval", [-1, True, 1.5])
def test_context_checkpoint_interval_requires_a_non_negative_integer(interval):
    config = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=64, tie_word_embeddings=True,
    )
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        ContextMemoryModel(
            Qwen3ForCausalLM(config), memory_length=4,
            memory_resampler_hidden_size=16, memory_resampler_heads=2,
            gradient_checkpoint_every_n_layers=interval,
        )


def test_zero_context_checkpoint_interval_disables_checkpointing(monkeypatch):
    model = tiny_model(
        context_gradient_checkpointing=True,
        gradient_checkpoint_every_n_layers=0,
    ).train()
    body = model._transformer_body
    enable = body.gradient_checkpointing_enable

    def fail_if_enabled(**kwargs):
        raise AssertionError("checkpointing must remain disabled")

    monkeypatch.setattr(body, "gradient_checkpointing_enable", fail_if_enabled)
    encode(model).memory.square().mean().backward()
    assert not any(layer.gradient_checkpointing for layer in body.layers)
    monkeypatch.setattr(body, "gradient_checkpointing_enable", enable)


def test_manual_lora_writer_forward_backward():
    model = tiny_model(context_gradient_checkpointing=True).train()
    prefix = encode(model)
    prefix.memory.square().mean().backward()
    assert model.context_resampler.latents.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for name, p in model.named_parameters() if "lora_B" in name)


def test_reader_passes_disable_lora_and_restores_state():
    model = tiny_model().eval()
    adapter = next(module for module in model.qwen.modules()
                   if module.__class__.__name__ == "MetaLoRA")
    seen = []
    handle = adapter.register_forward_pre_hook(
        lambda module, _inputs: seen.append(module.enabled)
    )
    try:
        prefix = encode(model, length=9, batch=1)
        question = model.qwen.get_input_embeddings()(torch.randint(0, 64, (1, 2)))
        answer = model.qwen.get_input_embeddings()(torch.randint(0, 64, (1, 2)))
        model.forward_with_memory_prefix(
            prefix, torch.zeros(1, dtype=torch.long), question,
            torch.ones(1, 2, dtype=torch.bool), answer,
            torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 2, dtype=torch.long),
        )
    finally:
        handle.remove()
    assert True in seen
    assert False in seen
    assert adapter.enabled is True


def test_each_lora_layer_is_enabled_only_for_encoder_and_disabled_for_reader():
    model = tiny_model().eval()
    adapters = {
        name: module
        for name, module in model.qwen.named_modules()
        if isinstance(module, MetaLoRA)
    }
    assert adapters
    seen = {name: [] for name in adapters}
    handles = [
        module.register_forward_pre_hook(
            lambda current, _inputs, name=name: seen[name].append(current.enabled)
        )
        for name, module in adapters.items()
    ]
    try:
        prefix = encode(model, length=9, batch=1)
        question = model.qwen.get_input_embeddings()(torch.randint(0, 64, (1, 2)))
        answer = model.qwen.get_input_embeddings()(torch.randint(0, 64, (1, 2)))
        model.forward_with_memory_prefix(
            prefix, torch.zeros(1, dtype=torch.long), question,
            torch.ones(1, 2, dtype=torch.bool), answer,
            torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 2, dtype=torch.long),
        )
    finally:
        for handle in handles:
            handle.remove()

    for name, states in seen.items():
        assert states, f"{name} was never called"
        assert True in states, f"{name} was not enabled for context encoding"
        assert False in states, f"{name} was not disabled for memory/reader forward"
        assert adapters[name].enabled is True, f"{name} did not restore its state"


def test_qa_has_no_task_embedding_and_recon_uses_one_embedding():
    model = tiny_model().eval()
    assert model.recon_task_embedding.shape == (model.qwen_hidden_size,)
    assert model.recon_task_embedding.requires_grad
    assert not hasattr(model, "task_embeddings")
    prefix = encode(model, length=9, batch=1)
    question = model.qwen.get_input_embeddings()(torch.randint(0, 64, (1, 2)))
    answer = model.qwen.get_input_embeddings()(torch.randint(0, 64, (1, 2)))
    common = dict(
        prefix=prefix, qa_context_indices=torch.zeros(1, dtype=torch.long),
        question_embeds=question, question_mask=torch.ones(1, 2, dtype=torch.bool),
        answer_embeds=answer, answer_mask=torch.ones(1, 2, dtype=torch.bool),
        labels=torch.ones(1, 2, dtype=torch.long), loss_only=False,
    )
    qa = model.forward_with_memory_prefix(**common)
    recon = model.forward_recon_with_memory_prefix(
        prefix, torch.zeros(1, dtype=torch.long), answer, torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.long),
    )
    assert recon.logits.shape[:2] == (1, 1 + 4 + 1 + 2 + 2)
    assert qa.logits.shape[:2] == (1, 4 + 2 + 2)
    assert recon.labels.shape == (1, 1 + 4 + 1 + 2 + 2)
    assert qa.labels.shape == (1, 4 + 2 + 2)
    recon.logits.float().square().mean().backward()
    assert model.recon_task_embedding.grad is not None
    assert model.recon_task_embedding.grad.abs().sum() > 0


def test_recon_reader_uses_memory_then_complete_user_and_assistant_turns():
    model = tiny_model().eval()
    prefix = encode(model, length=9, batch=2)
    embedding = model.qwen.get_input_embeddings()
    context = embedding(torch.randint(0, 64, (2, 2)))
    seen = []
    original = model.qwen.forward

    def capture(*args, **kwargs):
        seen.append(kwargs["inputs_embeds"].detach())
        return original(*args, **kwargs)

    model.qwen.forward = capture
    model.forward_recon_with_memory_prefix(
        prefix, torch.arange(2), context, torch.ones(2, 2, dtype=torch.bool),
        torch.ones(2, 2, dtype=torch.long),
    )
    assert len(seen) == 1
    torch.testing.assert_close(seen[0][:, :prefix.memory_length], prefix.memory_embedding)
    torch.testing.assert_close(
        seen[0][:, prefix.memory_length], embedding(torch.tensor(2)).expand(2, -1),
    )
    torch.testing.assert_close(
        seen[0][:, 1 + prefix.memory_length],
        model.recon_task_embedding.to(model.dtype).expand(2, -1),
    )
    torch.testing.assert_close(seen[0][:, 1 + prefix.memory_length + 1], embedding(torch.tensor(3)).expand(2, -1))
    torch.testing.assert_close(seen[0][:, 1 + prefix.memory_length + 2], embedding(torch.tensor(4)).expand(2, -1))


# Token ids returned for ``Reconstruct the context.`` by the test tokenizer.
RECON_TASK_TOKEN_IDS = (11, 22, 33)


def test_recon_task_embedding_is_initialised_from_user_instruction_mean():
    model = tiny_model(task_init_token_ids=RECON_TASK_TOKEN_IDS)
    embedding = model.qwen.get_input_embeddings().weight.detach()
    expected = embedding[list(RECON_TASK_TOKEN_IDS)].mean(dim=0)
    assert model.recon_task_embedding.shape == (model.qwen_hidden_size,)
    assert model.recon_task_embedding.dtype == torch.float32
    assert model.recon_task_embedding.requires_grad
    torch.testing.assert_close(model.recon_task_embedding.detach(), expected.to(torch.float32))
    # The marker is a copy of the instruction mean, not a view of the embedding.
    assert model.recon_task_embedding.data_ptr() != embedding.data_ptr()


def test_the_recon_marker_uses_the_ids_the_caller_resolves():
    """``load_model`` passes the ids from the run's own tokenizer; those rows are used."""
    model = tiny_model(task_init_token_ids=(3, 4))
    embedding = model.qwen.get_input_embeddings().weight.detach()
    assert model.task_init_token_ids == (3, 4)
    torch.testing.assert_close(
        model.recon_task_embedding.detach(),
        embedding[[3, 4]].mean(dim=0).to(torch.float32),
    )


def test_an_unusable_recon_phrase_id_uses_random_initialization():
    model = tiny_model(task_init_token_ids=(5, 10_000_000))
    assert model.task_init_token_ids == ()


def test_recon_phrase_ids_are_resolved_from_the_tokenizer():
    from src.model import recon_task_token_ids

    class _Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert text == "Reconstruct the context."
            return [11, 22, 33]

    class _Empty:
        def encode(self, text, add_special_tokens=False):
            return []

    assert recon_task_token_ids(_Tokenizer()) == (11, 22, 33)
    assert recon_task_token_ids(_Empty()) == ()
    assert recon_task_token_ids(object()) == ()


def test_recon_task_embedding_is_random_for_backbones_without_the_phrase_ids():
    """The guard is `max(id) < vocab`: missing phrase rows leave the marker random."""
    phrase_max = max(RECON_TASK_TOKEN_IDS)
    model = tiny_model(vocab_size=phrase_max)
    assert model.recon_task_embedding.shape == (model.qwen_hidden_size,)
    assert model.recon_task_embedding.requires_grad
    assert model.recon_task_embedding.dtype == torch.float32
    assert 0.005 < model.recon_task_embedding.std().item() < 0.05

    covered = tiny_model(vocab_size=phrase_max + 1, task_init_token_ids=RECON_TASK_TOKEN_IDS)
    embedding = covered.qwen.get_input_embeddings().weight.detach()
    torch.testing.assert_close(
        covered.recon_task_embedding.detach(),
        embedding[list(RECON_TASK_TOKEN_IDS)].mean(dim=0).to(torch.float32),
    )


def test_context_resampler_depth_and_block_order_are_configurable():
    model = tiny_model(memory_resampler_layers=3).eval()
    resampler = model.context_resampler
    assert resampler.num_layers == 3
    assert len(resampler.blocks) == 3
    assert all(block.self_attention is not None for block in resampler.blocks)
    assert all(block.cross_attention is not None for block in resampler.blocks)
    assert all(block.ffn is not None for block in resampler.blocks)


def test_context_resampler_self_attention_can_be_disabled():
    model = tiny_model(memory_resampler_use_self_attention=False)
    assert all(block.self_attention is None for block in model.context_resampler.blocks)
    assert all(block.cross_attention is not None for block in model.context_resampler.blocks)


def test_memory_embedding_is_a_single_sequence():
    model = tiny_model().eval()
    prefix = encode(model, length=9, batch=1)
    assert prefix.memory_embedding.shape == (1, model.context_resampler.num_memory,
                                             model.qwen_hidden_size)


def test_invalid_config_is_rejected_before_loading_tokenizer(monkeypatch):
    from transformers import AutoTokenizer

    def fail(*args, **kwargs):
        pytest.fail("tokenizer must not load for an invalid configuration")

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", fail)
    cfg = TrainConfig()
    cfg.memory.memory_resampler_layers = 0
    with pytest.raises(ValueError, match="resampler"):
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


def test_the_training_context_cap_is_stored_and_enforced():
    """``max_context_tokens`` is the *training* cap, and it is not silently dropped.

    The constructor used to validate the argument and then keep only the backbone window, so a
    caller could hand over any length and the model accepted it.
    """
    capped = tiny_model(max_context_tokens=8)
    assert capped.max_context_tokens == 8
    assert capped.window_limit == 64
    encode(capped, length=8)
    with pytest.raises(ValueError, match="max_context_tokens"):
        encode(capped, length=9)
    # Without an explicit cap the window is the cap, and the window itself is never exceeded.
    uncapped = tiny_model()
    assert uncapped.max_context_tokens == uncapped.window_limit == 64
    with pytest.raises(ValueError, match="window"):
        encode(uncapped, length=65)


def test_writer_config_roundtrip():
    config = TrainConfig.from_dict({"memory": {
        "context_layer_indices": [-1, -3], "question_resampler_info_source": "embedding",
    }, "data": {"max_context_tokens": 40960}})
    config.validate()
    assert TrainConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()
    for field, value in (("memory_resampler_hidden_size", 0), ("memory_resampler_ffn_ratio", 0),
                         ("memory_resampler_heads", 0), ("context_layer_indices", [0.5])):
        config = TrainConfig.from_dict({"memory": {field: value}})
        with pytest.raises(ValueError):
            config.validate()
