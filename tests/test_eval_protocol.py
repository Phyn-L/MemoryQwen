"""Evaluation-protocol invariants: the scored set must not depend on the number of GPUs.

The autoregressive budget (`evaluation.autoregressive_max_qa`) is a *global* number of QA
rows. It used to be a per-rank cap, so an 8-GPU run decoded ``8 x cap`` rows and a 4-GPU run
``4 x cap`` -- different question sets, different noise levels, and an A/B whose two arms were
launched with different rank counts silently stopped being comparable.

Run with ``python tests/test_eval_protocol.py`` or ``pytest tests``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.evaluator as evaluator_module  # noqa: E402
from src.data import QARecord  # noqa: E402
from src.evaluator import Evaluator, row_window  # noqa: E402
from src.model import MetaLoRA  # noqa: E402
from utils.config import TrainConfig  # noqa: E402

try:
    from transformers import Qwen3Config, Qwen3ForCausalLM
except ImportError:  # pragma: no cover - transformers is optional for the unit suite
    Qwen3Config = Qwen3ForCausalLM = None

VOCAB = 48
CONTEXT_LENGTH = 5
QUESTION_LENGTH = 3


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        from types import SimpleNamespace

        return SimpleNamespace(input_ids=[2] if text else [])

    def decode(self, ids, skip_special_tokens=True):
        return "cat"


def _available() -> bool:
    if Qwen3Config is None:
        print("   (skipped: transformers is not installed)")
        return False
    return True


def _model() -> MetaLoRA:
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=VOCAB, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, intermediate_size=32, tie_word_embeddings=True,
    )
    model = MetaLoRA(
        Qwen3ForCausalLM(config), rank=2, alpha=4.0, memory_length=2,
        decoder_hidden_size=8, decoder_heads=2, decoder_ffn_ratio=2,
        target_modules=["q_proj"], max_context_tokens=CONTEXT_LENGTH,
        trainable_dtype=torch.float32,
    )
    model.eval()
    return model


def _batches(rows_per_batch: int = 2, batches: int = 3) -> list[dict]:
    """A loader in global row order; every row's question identifies it."""
    out = []
    row = 0
    for _ in range(batches):
        records = [QARecord(f"q{row + i}", f"a{row + i}") for i in range(rows_per_batch)]
        out.append(
            {
                "context_ids": torch.randint(0, VOCAB, (1, CONTEXT_LENGTH)),
                "context_mask": torch.ones(1, CONTEXT_LENGTH, dtype=torch.bool),
                "question_ids": torch.randint(0, VOCAB, (rows_per_batch, QUESTION_LENGTH)),
                "question_mask": torch.ones(rows_per_batch, QUESTION_LENGTH, dtype=torch.bool),
                "qa_context_indices": torch.zeros(rows_per_batch, dtype=torch.long),
                "records": records,
            }
        )
        row += rows_per_batch
    return out


def _scored_rows(model, batches, max_qa, world=1, rank=0) -> list[str]:
    """Run the autoregressive pass and report which answers (i.e. rows) were scored."""
    cfg = TrainConfig()
    cfg.evaluation.max_new_tokens = 1
    cfg.evaluation.qa_batch_size = 2
    evaluator = Evaluator(_Tokenizer(), cfg)
    scored: list[str] = []
    original = evaluator._first_token_hit

    def record(token_id, record_):
        scored.append(record_.answer)
        return original(token_id, record_)

    evaluator._first_token_hit = record
    evaluator_module._distributed_world_size = lambda: world
    evaluator_module._distributed_rank = lambda: rank
    try:
        evaluator.autoregressive(
            model, batches, torch.device("cpu"), include_teacher_metrics=False,
            max_qa=max_qa, row_loader=batches,
        )
    finally:
        evaluator_module._distributed_world_size = _real_world
        evaluator_module._distributed_rank = _real_rank
    return scored


_real_world = evaluator_module._distributed_world_size
_real_rank = evaluator_module._distributed_rank


def test_row_window_tiles_the_global_budget_without_gaps_or_overlaps():
    for total in (1, 3, 6, 7, 1024, 1025):
        for world in (1, 2, 3, 4, 8):
            windows = [row_window(total, rank, world) for rank in range(world)]
            assert windows[0][0] == 0, (total, world, windows)
            for (start, end), (next_start, _) in zip(windows, windows[1:]):
                assert start <= end, (total, world, windows)
                assert end == next_start, f"gap or overlap at total={total} world={world}: {windows}"
            assert windows[-1][1] == total, (total, world, windows)
            # No rank is given more than ceil(total / world) rows.
            assert all(end - start <= -(-total // world) for start, end in windows)


def test_row_window_is_a_no_op_for_a_single_rank():
    assert row_window(10, 0, 1) == (0, 10)


def test_the_autoregressive_budget_is_global_not_per_rank():
    """`max_qa` counts rows across all ranks, so the scored set is rank-count invariant."""
    if not _available():
        return
    model = _model()
    batches = _batches()

    single = _scored_rows(model, batches, max_qa=3, world=1, rank=0)
    assert single == ["a0", "a1", "a2"], single

    # With three ranks the same budget must still score the same rows in total, one rank's
    # window each -- not three rows per rank.
    union = []
    for rank in range(3):
        union.extend(_scored_rows(model, batches, max_qa=3, world=3, rank=rank))
    assert union == ["a0", "a1", "a2"], union


def test_a_rank_only_scores_the_rows_inside_its_window():
    if not _available():
        return
    model = _model()
    batches = _batches(rows_per_batch=2, batches=3)      # rows 0..5 in global order
    assert _scored_rows(model, batches, max_qa=6, world=3, rank=0) == ["a0", "a1"]
    assert _scored_rows(model, batches, max_qa=6, world=3, rank=1) == ["a2", "a3"]
    assert _scored_rows(model, batches, max_qa=6, world=3, rank=2) == ["a4", "a5"]
    # A window that does not divide the batch size is still exact: the last rank of a
    # budget that stops mid-batch scores only the part inside its window.
    assert _scored_rows(model, batches, max_qa=5, world=2, rank=1) == ["a3", "a4"]


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
