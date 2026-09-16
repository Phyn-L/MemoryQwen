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


def _block_oracle(context_mask, memory_length, question_mask, answer_mask,
                  allow_slot_attention=False):
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
            if allow_slot_attention:
                for j in range(k + 1):
                    allowed[b, m0 + k, m0 + j] = True
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


def test_only_memory_rows_can_be_fully_blocked_and_only_with_an_empty_context():
    """The exact blocking contract, so a change in either direction is caught."""
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
        m0, q0 = context_len, context_len + memory_length

        # The generation path must never produce a fully blocked row.
        assert bool(continuation.any(dim=-1).all()), "continuation mask has a fully blocked row"

        for b in range(block.size(0)):
            for row in range(block.size(1)):
                fully_blocked = not bool(block[b, row].any())
                if m0 <= row < q0:
                    assert fully_blocked == (not context_mask[b].any()), (
                        f"memory row {row}: fully_blocked={fully_blocked} but that row has "
                        f"{int(context_mask[b].sum())} valid context token(s)"
                    )
                else:
                    assert not fully_blocked, (
                        f"row {row} (batch {b}) is fully blocked; only memory rows may be"
                    )


def test_slot_attention_only_adds_causal_memory_edges():
    """``allow_slot_attention`` changes memory rows and nothing else.

    Two things are pinned: the exact rule (slot k may read slots <= k), and that no
    other row's allowed set moves -- a leak into the question/answer rows would let the
    QA path see context, which is the invariant the whole memory design rests on.
    """
    for context_mask, question_mask, answer_mask, memory_length in _random_cases(seed=5):
        off = build_block_causal_mask(
            context_mask, memory_length, question_mask, answer_mask, torch.float32,
        )
        on = build_block_causal_mask(
            context_mask, memory_length, question_mask, answer_mask, torch.float32,
            allow_slot_attention=True,
        )
        got_off = _allowed(off)[0, 0]
        got_on = _allowed(on)[0, 0]
        assert torch.equal(got_off, _block_oracle(
            context_mask, memory_length, question_mask, answer_mask
        )[0]), "the default mask is no longer the documented one"
        assert torch.equal(got_on, _block_oracle(
            context_mask, memory_length, question_mask, answer_mask, allow_slot_attention=True
        )[0]), "slot attention diverged from the documented rule"

        context_len = context_mask.size(1)
        m0, q0 = context_len, context_len + memory_length
        outside_memory = torch.cat([
            got_off[:m0], got_off[q0:],
        ], dim=0)
        outside_memory_on = torch.cat([
            got_on[:m0], got_on[q0:],
        ], dim=0)
        assert torch.equal(outside_memory, outside_memory_on), (
            "slot attention changed a context or QA row"
        )


def test_blocked_entries_are_finite_so_no_row_can_nan():
    """Pin WHY a fully blocked memory row is safe: ``finfo.min``, not ``-inf``.

    Softmax over an all-``finfo.min`` row is uniform and finite; over an all-``-inf`` row it
    is NaN. Verified against the real Qwen3 attention with a fully blocked row: finite for
    both ``eager`` and ``sdpa`` with ``finfo.min``, NaN under ``eager`` with ``-inf``. So a
    "simplification" of the blocked value to ``-inf`` must fail this test.
    """
    for context_mask, question_mask, answer_mask, memory_length in _random_cases(seed=4, count=20):
        for dtype in (torch.float32, torch.bfloat16):
            mask = build_block_causal_mask(
                context_mask.bool(), memory_length, question_mask.bool(), answer_mask.bool(), dtype,
            )
            assert bool(torch.isfinite(mask).all()), "the block mask must contain no inf"
            # Realistic negative logits; blocked rows must stay finite through softmax.
            scores = torch.randn_like(mask) * 4.0 - 2.0
            weights = torch.softmax(scores + mask, dim=-1)
            assert bool(torch.isfinite(weights).all()), "softmax over the block mask was non-finite"

    # The failure mode guarded against, demonstrated directly.
    neg_inf_row = torch.full((1, 1, 1, 4), float("-inf"))
    assert not bool(torch.isfinite(torch.softmax(neg_inf_row, dim=-1)).all())


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
