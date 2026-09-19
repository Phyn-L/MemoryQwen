"""Reconstruction must not mutate a prefix reused by validation QA chunks."""
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from src.model import MetaLoRA


def test_reconstruction_preserves_prefix():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    torch.manual_seed(7)
    cfg = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8)
    model = MetaLoRA(Qwen3ForCausalLM(cfg).to(dtype=dtype), memory_length=4,
                     rank=2, decoder_hidden_size=8, decoder_heads=2,
                     max_context_tokens=12, embedding_recon=False, ae_lm=True).to(device)
    embed = model.qwen.get_input_embeddings()
    context = torch.randint(2,128,(2,12),device=device)
    question = torch.randint(2,128,(4,5),device=device)
    answer = torch.randint(2,128,(4,3),device=device)
    indices = torch.tensor([0,0,1,1],device=device)
    for training in (False, True):
        model.train(training)
        with torch.set_grad_enabled(training):
            prefix = model.encode_context_prefix(embed(context), context.bool())
            saved = [(layer.keys.clone(), layer.values.clone()) for layer in prefix.memory_cache.layers]
            baseline = model.forward_qa_with_prefix(prefix, indices, embed(question), question.bool(),
                                                    embed(answer), answer.bool(), answer)
            recon = model.autoencode_with_memory(prefix, embed(context), context.bool())
            assert prefix.memory_cache.get_seq_length() == 4
            for layer, (keys, values) in zip(prefix.memory_cache.layers, saved):
                torch.testing.assert_close(layer.keys, keys)
                torch.testing.assert_close(layer.values, values)
            for _ in range(2):
                output = model.forward_qa_with_prefix(prefix, indices, embed(question), question.bool(),
                                                      embed(answer), answer.bool(), answer)
                torch.testing.assert_close(output.logits, baseline.logits)
            if training:
                (output.logits.float().square().mean() + recon.float().square().mean()).backward()
                assert model.memory_tokens.grad is not None
                assert torch.isfinite(model.memory_tokens.grad).all()
                assert model.memory_tokens.grad.abs().sum() > 0
    print(f'cache isolation, repeated QA and backward passed on {device}, {dtype}')
