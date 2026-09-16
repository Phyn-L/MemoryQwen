"""Metric-stack contract.

Two things are easy to get wrong and are asserted here:

1. The evaluator reduces a *flattened list* of metric accumulators through
   ``dist.all_reduce``. If a key were missing from ``METRIC_KEYS``, or ordered differently
   on different ranks, the reduction would silently mix values between metrics. So the
   evaluator's key list must be exactly the key set that ``qa_metrics_all`` produces, in
   the same order.
2. The ``_official`` metrics must really be the official-normalized ones, and the
   unsuffixed ones must really be the legacy-normalized ones -- otherwise the whole point
   of reporting both (comparing with the ICL baseline without invalidating old runs) is
   lost.

Also documents the divergence the two rulers have, which is why both exist.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import (  # noqa: E402
    LEGACY_METRIC_KEYS,
    METRIC_KEYS,
    OFFICIAL_SUFFIX,
    normalize_official,
    normalize_text,
    qa_metrics,
    qa_metrics_all,
    qa_metrics_official,
)

SAMPLES = [
    ("a cat", "cat"),
    ("the cat", "a cat"),
    ("cat", "cat"),
    ("Paris", ["paris"][0]),
    ("hello, world!", "hello world"),
    ("", ""),
    ("", "answer"),
    ("answer", ""),
    ("New York City", "new york city"),
    ("one two three four", "one two three"),
]


def test_metric_keys_match_what_the_evaluator_produces():
    produced = tuple(qa_metrics_all("a cat", "cat").keys())
    assert produced == METRIC_KEYS, f"evaluator key order {METRIC_KEYS} != produced {produced}"


def test_metric_keys_are_two_disjoint_normalizations_of_the_same_four_metrics():
    assert len(set(METRIC_KEYS)) == len(METRIC_KEYS), "duplicate key in METRIC_KEYS"
    assert METRIC_KEYS == LEGACY_METRIC_KEYS + tuple(
        f"{key}{OFFICIAL_SUFFIX}" for key in LEGACY_METRIC_KEYS
    )


def test_official_keys_are_the_official_normalization():
    for prediction, reference in SAMPLES:
        combined = qa_metrics_all(prediction, reference)
        expected = qa_metrics(prediction, reference, normalize_official)
        for key, value in expected.items():
            assert combined[f"{key}{OFFICIAL_SUFFIX}"] == value, (prediction, reference, key)


def test_legacy_keys_are_unchanged_by_this_feature():
    """The unsuffixed keys must stay on the training-time ruler, byte for byte."""
    for prediction, reference in SAMPLES:
        combined = qa_metrics_all(prediction, reference)
        legacy = qa_metrics(prediction, reference)
        for key, value in legacy.items():
            assert combined[key] == value, (prediction, reference, key)


def test_the_two_rulers_really_do_differ():
    """Guards the reason both sets exist: if they agreed, one set would be redundant."""
    prediction, reference = "a cat", "cat"
    assert not normalize_text(prediction) == normalize_official(prediction)
    legacy = qa_metrics(prediction, reference)
    official = qa_metrics_official(prediction, reference)
    assert legacy["em"] == 0.0 and legacy["f1"] < 1.0
    assert official[f"em{OFFICIAL_SUFFIX}"] == 1.0 and official[f"f1{OFFICIAL_SUFFIX}"] == 1.0


def test_official_matches_the_icl_baseline_stack():
    """The ICL baseline's own metric functions must agree with our _official keys."""
    from src.icl_baseline import example_metrics

    for prediction, reference in SAMPLES:
        baseline = example_metrics(prediction, [reference])
        official = qa_metrics_official(prediction, reference)
        for key in ("em", "f1", "rouge_l"):
            assert baseline[key] == official[f"{key}{OFFICIAL_SUFFIX}"], (prediction, reference, key)


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
