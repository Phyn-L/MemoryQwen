"""Metric-stack contract.

The evaluator must produce the *same* number for the same prediction.
They previously differed in the normalizer (articles kept vs dropped) and in the reduction
over several gold answers (first reference vs max over all). Both are now single-sourced,
and this file pins that down:

1. ``METRIC_KEYS`` is the evaluator's accumulator order. A distributed reduce flattens the
   accumulators into a list, so a key missing from that tuple -- or ordered differently on
   different ranks -- would silently mix values between metrics.
2. There is exactly one normalizer, and it is the official SQuAD one.
3. ``best_reference_metrics`` is the metric-wise max over references.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.eval_metrics import (  # noqa: E402
    METRIC_KEYS,
    best_reference_metrics,
    normalize_answer,
    qa_metrics,
)


def test_metric_keys_match_what_the_evaluator_produces():
    produced = tuple(qa_metrics("a cat", "cat").keys())
    assert produced == METRIC_KEYS, f"evaluator order {METRIC_KEYS} != produced {produced}"


def test_there_is_one_normalizer_and_it_is_the_official_one():
    # Articles are dropped, punctuation is removed, everything is lowercased.
    assert normalize_answer("The  Cat, sat!") == "cat sat"
    assert normalize_answer("A CAT") == normalize_answer("cat")
    # The old training-time normalizer kept articles; that divergence is gone.
    assert qa_metrics("a cat", "cat")["em"] == 1.0
    assert qa_metrics("the cat", "a cat")["em"] == 1.0
    assert qa_metrics("a cat", "cat")["f1"] == 1.0


def test_best_reference_metrics_is_the_metric_wise_max():
    prediction, references = "a dog", ["cat", "a dog", "the dog"]
    got = best_reference_metrics(prediction, references)
    per_reference = [qa_metrics(prediction, reference) for reference in references]
    for key in METRIC_KEYS:
        assert got[key] == max(row[key] for row in per_reference), key
        # Every reference contributes at least one metric-wise maximum here, so a
        # first-reference-only reduction would be strictly worse.
    assert got["em"] == 1.0


def test_empty_reference_list_is_treated_as_one_empty_answer():
    assert best_reference_metrics("cat", []) == qa_metrics("cat", "")


def test_unigram_precision_is_gone():
    """``precision`` was dropped: short answers gamed it and it tracked ``f1``.

    It also changes the flattened layout of the distributed reduce, so removing it is not
    a cosmetic change and is asserted rather than left to review.
    """
    assert "precision" not in METRIC_KEYS
    assert "precision" not in qa_metrics("a cat", "cat")


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
