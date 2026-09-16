"""Two-dtype contract for the LoRA implementations (P2).

Run with either ``python tests/test_lora_dtype.py`` or ``pytest tests``.

What is asserted
----------------
1. ``StaticLoRALinear`` (the default backend) really computes its LoRA delta in
   float32 even when the surrounding forward runs under bf16 autocast. The test
   is discriminating: it also checks that the same computation *without*
   ``no_autocast`` would be off by far more than the tolerance.
2. ``MetaLoRA`` defaults to the static backend regardless of whether ``peft`` is
   installed, and records which backend it chose.
3. ``disable_autocast_for_peft_lora`` wraps PEFT-style layers (module exposing
   ``lora_A``/``lora_B``), turns autocast off inside them, leaves their output
   untouched, and is idempotent.
"""
from __future__ import annotations

from types import SimpleNamespace
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import (  # noqa: E402
    MetaLoRA,
    StaticLoRALinear,
    disable_autocast_for_peft_lora,
    is_trainable_parameter_name,
)

TOL = 1e-6


def _static_layer(dim=8, out=4, rank=2, alpha=4.0):
    torch.manual_seed(0)
    base = nn.Linear(dim, out, bias=False).to(torch.bfloat16)
    layer = StaticLoRALinear(base, rank=rank, alpha=alpha, dropout=0.0, dtype=torch.float32)
    # lora_B is zero-initialised, so the delta would be exactly 0 and the dtype
    # contract untestable. Give both matrices real values.
    with torch.no_grad():
        layer.lora_A.normal_(std=0.3)
        layer.lora_B.normal_(std=0.3)
    return base, layer


def _reference_forward(base, layer, x):
    """Exact contract of StaticLoRALinear.forward.

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

    # Discriminating check: computing the delta under autocast (i.e. what the PEFT
    # branch would do without the wrapper) rounds every intermediate to bf16.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        rounded = (
            base(x.to(torch.bfloat16))
            + layer.scaling * (x.to(torch.bfloat16) @ layer.lora_A.t() @ layer.lora_B.t())
        ).float()
    discrimination = (rounded - expected).abs().max().item()
    assert discrimination > 1e-4, (
        f"test does not discriminate: autocast path differs by only {discrimination}"
    )


def _cpu_autocast_enabled() -> bool:
    """torch.is_autocast_enabled() defaults to the CUDA flag; we run on CPU here."""
    try:
        return bool(torch.is_autocast_enabled("cpu"))
    except TypeError:  # older torch
        return bool(torch.is_autocast_cpu_enabled())


class _FakeLoraLayer(nn.Module):
    """Stand-in for a PEFT LoRA layer: exposes lora_A/lora_B and reports autocast."""

    def __init__(self, dim=8):
        super().__init__()
        torch.manual_seed(0)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(dim, 2, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(2, dim, bias=False)})
        self.seen_autocast: list[bool] = []

    def forward(self, x):
        self.seen_autocast.append(_cpu_autocast_enabled())
        return self.lora_A["default"](x)


def test_disable_autocast_wrapper_wraps_and_disables():
    layer = _FakeLoraLayer()
    x = torch.randn(4, 8)

    reference = layer(x)  # no autocast anywhere: the float32 result
    with torch.autocast("cpu", dtype=torch.bfloat16):
        layer(x)
    assert layer.seen_autocast[-1] is True, "autocast should be visible before wrapping"

    assert disable_autocast_for_peft_lora(layer) == 1
    with torch.autocast("cpu", dtype=torch.bfloat16):
        wrapped = layer(x)

    assert layer.seen_autocast[-1] is False, "autocast was still enabled inside the wrapped forward"
    assert torch.allclose(reference, wrapped, atol=TOL), "wrapper changed the result"


def test_disable_autocast_wrapper_is_idempotent():
    layer = _FakeLoraLayer()
    assert disable_autocast_for_peft_lora(layer) == 1
    assert disable_autocast_for_peft_lora(layer) == 0


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


def _fake_metaloRA(use_peft=False):
    return MetaLoRA(
        _FakeQwen(),
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
        use_peft=use_peft,
    )


def test_metaloRA_defaults_to_static_backend():
    model = _fake_metaloRA()
    assert model.lora_backend == "static"
    assert any(isinstance(module, StaticLoRALinear) for module in model.qwen.modules())


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
    assert "memory_tokens" in trainable
    assert any(name.startswith("decoders.") for name in trainable)


def test_use_peft_true_without_peft_raises_or_uses_peft():
    try:
        import peft  # noqa: F401
    except ImportError:
        try:
            _fake_metaloRA(use_peft=True)
        except RuntimeError as error:
            assert "peft" in str(error)
        else:
            raise AssertionError("use_peft=True without peft installed must raise, not fall back silently")
    else:
        model = _fake_metaloRA(use_peft=True)
        assert model.lora_backend == "peft"


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
