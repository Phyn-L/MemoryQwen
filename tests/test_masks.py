"""Attention visibility for complete memory-prefixed embedding sequences."""
import pytest
import torch

from src.model import build_causal_mask


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_complete_sequence_causal_visibility(dtype):
    valid = torch.tensor([[True, True, False, True]])
    mask = build_causal_mask(valid, dtype)[:, 0]
    expected = torch.tensor([[True, False, False, False],
                             [True, True, False, False],
                             [False, False, True, False],
                             [True, True, False, True]])
    assert torch.equal(mask[0] == 0, expected)
    assert torch.isfinite(mask).all()
