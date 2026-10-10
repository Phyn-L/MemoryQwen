"""Independent oracle for the short memory-prefixed QA attention mask."""
import pytest
import torch

from src.model import build_continuation_mask


@pytest.mark.parametrize("readout", [0, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_continuation_visibility(readout, dtype):
    generator = torch.Generator().manual_seed(42)
    for _ in range(40):
        q, a, m = torch.randint(1, 7, (3,), generator=generator).tolist()
        question = torch.rand(2, q, generator=generator) > 0.4
        answer = torch.rand(2, a, generator=generator) > 0.4
        mask = build_continuation_mask(question, answer, m, dtype, readout)
        expected = torch.zeros(2, q + readout + a, m + q + readout + a, dtype=torch.bool)
        for b in range(2):
            for row in range(q + readout + a):
                if row < q:
                    valid = question[b, row]
                    visible = list(range(m)) + [m + j for j in range(row + 1) if question[b, j]]
                elif row < q + readout:
                    valid = True
                    visible = list(range(m)) + [m + j for j in range(q) if question[b, j]]
                else:
                    i = row - q - readout
                    valid = answer[b, i]
                    visible = (
                        list(range(m)) + [m + j for j in range(q) if question[b, j]]
                        + list(range(m + q, m + q + readout))
                        + [m + q + readout + j for j in range(i + 1) if answer[b, j]]
                    )
                expected[b, row, visible if valid else [m + row]] = True
        assert torch.equal(mask[:, 0] == 0, expected)
        assert expected.any(dim=-1).all()
        assert torch.isfinite(mask).all()
