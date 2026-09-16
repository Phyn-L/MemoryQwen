"""Attention-mask contract (P4).

Why this file exists
--------------------
``build_continuation_mask`` is a strict restriction of ``build_block_causal_mask``: with
an empty context, the block mask's question/answer rows are exactly the continuation
mask. Deriving one from the other would remove ~30 lines but makes the hot path build a
larger mask (measured 0.21 ms -> 0.28 ms per call at B=4, Q+A=256, M=8) and compute memory
rows that are then thrown away. The two builders are therefore kept separate and pinned
together here, so a rule change in one cannot silently diverge from the other.

The oracle below is an independent, row-by-row transcription of the documented attention
rules. It is deliberately not the implementation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import build_block_causal_mask, build_continuation_mask  # noqa: E402

MIN = torch.finfo(torch.float32).min


def _allowed(mask: torch.Tensor) -> torch.Tensor:
    """masked_fill writes finfo.min for blocked entries; everything else is exactly 0."""
    return mask == 0


def _block_oracle(context_mask, memory_length, question_mask, answer_mask):
    """Independent transcription of the block-mask rules. Returns [B, T, T] bool."""
    context_mask, question_mask, answer_mask = (
        context_mask.bool(), question_mask.bool(), answer_mask.bool()
    )
    bsz, context_len = context_mask.shape
    question_len, answer_len = question_mask.shape[1], answer_mask.shape[1]
    total = context_len + memory_length + question_len + answer_len
    m0, q0 = context_len, context_len + memory_length
    a0 = q0 + question_len
    allowed = torch.zeros(bsz, total, total, dtype=torch.bool)
    for b in range(bsz):
        for i in range(context_len):
            if context_mask[b, i]:
                for j in range(i + 1):
                    allowed[b, i, j] = bool(context_mask[b, j])
            else:
                allowed[b, i, i] = True
        for k in range(memory_length):
            for j in range(context_len):
                allowed[b, m0 + k, j] = bool(context_mask[b, j])
        for i in range(question_len):
            if question_mask[b, i]:
                for j in range(m0, q0):
                    allowed[b, q0 + i, j] = True
                for j in range(i + 1):
                    allowed[b, q0 + i, q0 + j] = bool(question_mask[b, j])
            else:
                allowed[b, q0 + i, q0 + i] = True
        for i in range(answer_len):
            if answer_mask[b, i]:
                for j in range(m0, q0):
                    allowed[b, a0 + i, j] = True
                for j in range(question_len):
                    allowed[b, a0 + i, q0 + j] = bool(question_mask[b, j])
                for j in range(i + 1):
                    allowed[b, a0 + i, a0 + j] = bool(answer_mask[b, j])
            else:
                allowed[b, a0 + i, a0 + i] = True
    return allowed


def _continuation_oracle(question_mask, answer_mask, memory_length):
    question_mask, answer_mask = question_mask.bool(), answer_mask.bool()
    bsz, question_len = question_mask.shape
    answer_len = answer_mask.shape[1]
    current = question_len + answer_len
    allowed = torch.zeros(bsz, current, memory_length + current, dtype=torch.bool)
    for b in range(bsz):
        for i in range(question_len):
            if question_mask[b, i]:
                allowed[b, i, :memory_length] = True
                for j in range(i + 1):
                    allowed[b, i, memory_length + j] = bool(question_mask[b, j])
            else:
                allowed[b, i, memory_length + i] = True
        for i in range(answer_len):
            row = question_len + i
            if answer_mask[b, i]:
                allowed[b, row, :memory_length] = True
                for j in range(question_len):
                    allowed[b, row, memory_length + j] = bool(question_mask[b, j])
                for j in range(i + 1):
                    allowed[b, row, memory_length + question_len + j] = bool(answer_mask[b, j])
            else:
                allowed[b, row, memory_length + row] = True
    return allowed


def _random_cases(count=120, seed=0):
    generator = torch.Generator().manual_seed(seed)

    def bools(b, n, keep):
        return torch.rand(b, n, generator=generator) < keep

    for _ in range(count):
        b = int(torch.randint(1, 3, (1,), generator=generator))
        c = int(torch.randint(0, 9, (1,), generator=generator))
        q = int(torch.randint(0, 6, (1,), generator=generator))
        a = int(torch.randint(0, 6, (1,), generator=generator))
        keep = float(torch.rand(1, generator=generator)) * 0.9 + 0.05
        yield (
            bools(b, c, keep), bools(b, q, keep), bools(b, a, keep),
            int(torch.randint(1, 4, (1,), generator=generator)),
        )


def _as_min(mask: torch.Tensor) -> torch.Tensor:
    return mask.masked_fill(~mask, MIN) if mask.dtype == torch.bool else mask


def test_block_mask_matches_independent_oracle():
    for context_mask, question_mask, answer_mask, memory_length in _random_cases():
        got = _allowed(build_block_causal_mask(context_mask, memory_length, question_mask, answer_mask, torch.float32))[0, 0]
        want = _block_oracle(context_mask, memory_length, question_mask, answer_mask)[0]
        assert torch.equal(got, want), "block mask diverged from the documented rules"


def test_continuation_mask_matches_independent_oracle():
    for _, question_mask, answer_mask, memory_length in _random_cases(seed=1):
        got = _allowed(build_continuation_mask(question_mask, answer_mask, memory_length, torch.float32))[0, 0]
        want = _continuation_oracle(question_mask, answer_mask, memory_length)[0]
        assert torch.equal(got, want), "continuation mask diverged from the documented rules"


def test_continuation_mask_is_block_mask_with_empty_context():
    """The invariant that makes it safe to keep two builders."""
    for context_mask, question_mask, answer_mask, memory_length in _random_cases(seed=2):
        bsz = question_mask.size(0)
        empty_context = torch.zeros(bsz, 0, dtype=torch.bool)
        derived = build_block_causal_mask(
            empty_context, memory_length, question_mask, answer_mask, torch.float32,
        )[:, :, memory_length:, :]
        direct = build_continuation_mask(question_mask, answer_mask, memory_length, torch.float32)
        assert torch.equal(derived, direct), "continuation mask is no longer the empty-context block mask"


def test_no_valid_row_is_fully_masked():
    """Every real query row must keep at least one unmasked key, or softmax yields NaN."""
    for context_mask, question_mask, answer_mask, memory_length in _random_cases(seed=3):
        context_mask, question_mask, answer_mask = (
            context_mask.bool(), question_mask.bool(), answer_mask.bool()
        )
        block = _allowed(
            build_block_causal_mask(context_mask, memory_length, question_mask, answer_mask, torch.float32)
        )[0]
        continuation = _allowed(
            build_continuation_mask(question_mask, answer_mask, memory_length, torch.float32)
        )[0]

        context_len = context_mask.size(1)
        m0 = context_len
        q0 = context_len + memory_length

        # The continuation mask (generation path) must never have an empty row.
        assert bool(continuation.any(dim=-1).all()), "continuation mask has an all-(-inf) row"

        # Block mask: context rows (real or padded) and QA rows (real or padded) always
        # have a key. Memory rows [m0, q0) are the documented exception: they only see
        # context, so an all-padded context masks them completely. The data pipeline drops
        # empty contexts, which is what keeps that unreachable.
        for b in range(block.size(0)):
            for row in range(block.size(1)):
                if m0 <= row < q0:
                    if not context_mask[b].any():
                        assert not bool(block[b, row].any()), (
                            "memory row should be fully masked when the context is empty"
                        )
                    continue
                assert bool(block[b, row].any()), (
                    f"block mask row {row} (batch {b}) is all -inf although the query always has a key"
                )


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
