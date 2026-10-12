import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.model import ContextMemoryModel


def test_empty_lora_preserves_memory_gradient():
    config = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    model = ContextMemoryModel(Qwen3ForCausalLM(config), target_modules=[], memory_length=4, max_context_tokens=16)
    assert model.lora_backend == "none"
    ids = torch.randint(0, 64, (1, 8))
    prefix = model.encode_context_prefix(model.qwen.get_input_embeddings()(ids), ids.bool())
    prefix.memory.square().mean().backward()
    assert model.context_resampler.latents.grad is not None
    assert model.context_resampler.latents.grad.abs().sum() > 0
