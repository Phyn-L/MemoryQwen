"""Reconstruction must not mutate memory embeddings reused by validation QA chunks."""
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.model import ContextMemoryModel


def test_reconstruction_preserves_memory_embeddings():
    # Deliberately CPU: this micro-model gains nothing from a GPU, and picking one whenever
    # torch.cuda.is_available() made the test fail with "CUDA out of memory" on a shared box.
    device = torch.device('cpu')
    dtype = torch.float32
    torch.manual_seed(7)
    cfg = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8)
    model = ContextMemoryModel(Qwen3ForCausalLM(cfg).to(dtype=dtype), memory_length=4,
                     rank=2, max_context_tokens=12,
                     recon_user_token_ids=(2,), recon_answer_prefix_token_ids=(3, 4)).to(device)
    embed = model.qwen.get_input_embeddings()
    context = torch.randint(2,128,(2,12),device=device)
    question = torch.randint(2,128,(4,5),device=device)
    answer = torch.randint(2,128,(4,3),device=device)
    indices = torch.tensor([0,0,1,1],device=device)
    for training in (False, True):
        model.train(training)
        with torch.set_grad_enabled(training):
            prefix = model.encode_context_prefix(embed(context), context.bool())
            saved = prefix.memory_embedding.clone()
            baseline = model.forward_with_memory_prefix(prefix, indices, embed(question), question.bool(),
                                                    embed(answer), answer.bool(), answer)
            model.recon_teacher_forced(prefix, embed(context), context.bool(), context)
            torch.testing.assert_close(prefix.memory_embedding, saved)
            for _ in range(2):
                output = model.forward_with_memory_prefix(prefix, indices, embed(question), question.bool(),
                                                      embed(answer), answer.bool(), answer)
                torch.testing.assert_close(output.logits, baseline.logits)
            if training:
                output.logits.float().square().mean().backward()
                assert model.context_resampler.latents.grad is not None
                assert torch.isfinite(model.context_resampler.latents.grad).all()
                assert model.context_resampler.latents.grad.abs().sum() > 0
    print(f'memory embedding isolation, repeated QA and backward passed on {device}, {dtype}')
