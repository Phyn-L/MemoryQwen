from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("test_recon_script", ROOT / "scripts/test_recon.py")
test_recon = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(test_recon)


def test_select_contexts_takes_two_squad_and_marco_rows_in_manifest_order():
    manifest = {
        "datasets": {
            "squad": {"contexts": [{"context_id": f"s{i}", "context": f"squad {i}"} for i in range(3)]},
            "msmarcov1.1": {"contexts": [{"context_id": f"m{i}", "context": f"marco {i}"} for i in range(3)]},
        }
    }

    examples = test_recon.select_contexts(manifest)

    assert [(row["dataset"], row["context_id"]) for row in examples] == [
        ("squad", "s0"), ("squad", "s1"), ("msmarcov1.1", "m0"), ("msmarcov1.1", "m1")
    ]


def test_select_contexts_requires_both_datasets_and_nonempty_contexts():
    with pytest.raises(ValueError, match="need 2"):
        test_recon.select_contexts({"datasets": {"squad": {"contexts": []}}})

    manifest = {
        "datasets": {
            "squad": {"contexts": [{"context": ""}, {"context": "ok"}]},
            "msmarcov1.1": {"contexts": [{"context": "m1"}, {"context": "m2"}]},
        }
    }
    with pytest.raises(ValueError, match="empty squad context"):
        test_recon.select_contexts(manifest)


def test_build_report_contains_example_comparisons_and_dataset_averages():
    examples = [
        {"dataset": "squad", "context_id": "s1", "context": "alpha beta"},
        {"dataset": "msmarcov1.1", "context_id": "m1", "context": "gamma delta"},
    ]

    report = test_recon.build_report(examples, ["alpha beta", "gamma delta"], ["alpha beta", "gamma"])

    assert report["examples"][0]["metrics"]["em"] == 1.0
    assert report["metrics"]["dataset"]["squad"]["f1"] == 1.0
    assert report["metrics"]["dataset"]["msmarcov1.1"]["f1"] == pytest.approx(2 / 3)
    assert report["metrics"]["overall"]["em"] == 0.5
