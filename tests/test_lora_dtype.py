"""Two-dtype contract for the LoRA implementations (P2).

Run with either ``python tests/test_lora_dtype.py`` or ``pytest tests``.

What is asserted
----------------
1. ``MetaLoRA`` (the default backend) really computes its LoRA delta in
   float32 even when the surrounding forward runs under bf16 autocast. The test
   is discriminating: it also checks that the same computation *without*
   ``no_autocast`` would be off by far more than the tolerance.
2. ``ContextMemoryModel`` uses the hand-written MetaLoRA layers.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import (  # noqa: E402
    ContextMemoryModel,
    MetaLoRA,
    is_trainable_parameter_name,
)

TOL = 1e-6


def _static_layer(dim=8, out=4, rank=2, alpha=4.0):
    torch.manual_seed(0)
    base = nn.Linear(dim, out, bias=False).to(torch.bfloat16)
    layer = MetaLoRA(base, rank=rank, alpha=alpha, dropout=0.0, dtype=torch.float32)
    # lora_B is zero-initialised, so the delta would be exactly 0 and the dtype
    # contract untestable. Give both matrices real values.
    with torch.no_grad():
        layer.lora_A.normal_(std=0.3)
        layer.lora_B.normal_(std=0.3)
    return base, layer


def _reference_forward(base, layer, x):
    """Exact contract of MetaLoRA.forward.

    The base matmul runs in the backbone dtype, the LoRA delta is computed in the
    trainable dtype (float32), and only the finished delta is rounded back to the
    backbone dtype before being added.
    """
    base_out = base(x.to(torch.bfloat16))
    lora_x = x.to(torch.float32)
    delta = layer.scaling * (lora_x @ layer.lora_A.t().float() @ layer.lora_B.t().float())
    return (base_out + delta.to(base.weight.dtype)).float()


def test_static_lora_keeps_fp32_parameters():
    _, layer = _static_layer()
    assert layer.lora_A.dtype == torch.float32
    assert layer.lora_B.dtype == torch.float32


def test_static_lora_delta_is_fp32_under_autocast():
    base, layer = _static_layer()
    x = torch.randn(5, 8, dtype=torch.float32)

    expected = _reference_forward(base, layer, x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        got = layer(x)

    error = (got.float() - expected).abs().max().item()
    assert error < TOL, f"static LoRA did not follow the fp32 contract: max abs error {error}"

    # Discriminating check: computing the delta under autocast without the explicit dtype
    # boundary rounds every intermediate to bf16.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        rounded = (
            base(x.to(torch.bfloat16))
            + layer.scaling * (x.to(torch.bfloat16) @ layer.lora_A.t() @ layer.lora_B.t())
        ).float()
    discrimination = (rounded - expected).abs().max().item()
    assert discrimination > 1e-4, (
        f"test does not discriminate: autocast path differs by only {discrimination}"
    )


class _FakeQwen(nn.Module):
    """Minimal stand-in for a Qwen backbone (no transformers needed)."""

    def __init__(self, dim=16, layers=2, vocab=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(
            [nn.ModuleDict({"q_proj": nn.Linear(dim, dim, bias=False)}) for _ in range(layers)]
        )
        self.config = SimpleNamespace(hidden_size=dim, num_hidden_layers=layers, max_position_embeddings=128)

    def get_input_embeddings(self):
        return self.embed


def _fake_metaloRA():
    return ContextMemoryModel(
        _FakeQwen(),
        rank=2,
        alpha=4.0,
        memory_length=2,
                                target_modules=["q_proj"],
        max_context_tokens=8,
        trainable_dtype=torch.float32,
        recon_user_token_ids=(2,), recon_answer_prefix_token_ids=(3, 4),
            )


def test_manual_metaloRA_backend():
    model = _fake_metaloRA()
    assert model.lora_backend == "manual"
    assert any(isinstance(module, MetaLoRA) for module in model.qwen.modules())


def test_trainable_parameters_are_fp32_and_flagged():
    model = _fake_metaloRA()
    for name, parameter in model.named_parameters():
        if is_trainable_parameter_name(name):
            assert parameter.dtype == torch.float32, f"{name} is {parameter.dtype}"
            parameter.requires_grad = True
        else:
            parameter.requires_grad = False
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable, "no trainable parameters found"
    assert all(is_trainable_parameter_name(name) for name in trainable)
    assert any("lora_A" in name for name in trainable)
    assert "context_resampler.latents" in trainable
    assert any(name.startswith("context_layer_logits") for name in trainable)


def test_trainable_parameter_dtypes_counts_by_requires_grad():
    """The introspector must agree with the single trainable-name rule."""
    model = _fake_metaloRA()
    for name, parameter in model.named_parameters():
        parameter.requires_grad = is_trainable_parameter_name(name)
    counts = model.trainable_parameter_dtypes()
    assert set(counts) == {"torch.float32"}, counts
    assert sum(counts.values()) == len([p for p in model.parameters() if p.requires_grad])


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
            except Exception as error:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(error).__name__}: {error}")
            else:
                print(f"ok   {name}")
    print("\nVERDICT:", "ALL PASSED" if not failures else f"{failures} FAILED")
    raise SystemExit(1 if failures else 0)
