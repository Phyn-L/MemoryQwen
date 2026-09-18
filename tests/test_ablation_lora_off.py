import torch
from tests.test_tied_backbone import _tiny_qwen3, _batch, _forward_losses
from src.model import MetaLoRA


def test_empty_targets_disable_lora_but_preserve_memory_gradient():
    model = MetaLoRA(_tiny_qwen3(), target_modules=[], memory_length=4,
                     decoder_hidden_size=16, decoder_heads=4,
                     max_context_tokens=16, context_lm=True)
    assert model.lora_backend == "none"
    assert not any("lora_" in n for n, _ in model.named_parameters())
    assert not any(p.requires_grad for p in model.qwen.parameters())
    _, qa, rec = _forward_losses(model, _batch())
    (qa + rec).backward()
    assert torch.isfinite(model.memory_tokens.grad).all()
    assert model.memory_tokens.grad.abs().sum() > 0
