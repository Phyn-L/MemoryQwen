"""Autoregressive evaluation logging contract."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("_train_script", REPO / "scripts/train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def test_generated_answer_metrics_and_dataset_metrics_are_logged():
    metrics = {key: 0.7 for key in train.ANSWER_METRIC_KEYS}
    metrics.update({
        "dataset/squad/em": 0.8,
        "dataset/squad/f1": 0.7,
        "dataset/squad/rouge_l": 0.6,
        "dataset/squad/count": 256,
        "dataset/hotpotqa(distractor)/f1": 0.5,
    })
    payload, = train.eval_log_payloads({"qa_task": metrics, "recon_task": metrics})
    for task_name in ("qa_task", "recon_task"):
        prefix = f"val_autoregressive_{task_name}/"
        assert payload[prefix + "squad_EM"] == 0.8
        assert payload[prefix + "squad_F1"] == 0.7
        assert payload[prefix + "squad_ROUGE-L"] == 0.6
        assert payload[prefix + "squad_count"] == 256
        assert payload[prefix + "hotpotqa_distractor_F1"] == 0.5
        for key in train.ANSWER_METRIC_KEYS:
            assert payload[prefix + key] == 0.7
    assert all(key.startswith("val_autoregressive_") for key in payload)


def test_evaluation_does_not_log_losses_or_empty_results():
    assert train.eval_log_payloads() == []
    values = {"f1": 0.5, "loss": 1.0, "qa_loss": 2.0, "ppl": 3.0}
    assert train.eval_log_payloads({"qa_task": values}) == [
        {"val_autoregressive_qa_task/f1": 0.5}
    ]


def test_eval_metrics_table_groups_datasets_and_metrics():
    table = train.format_eval_metrics_table(
        {
            "qa_task": {
                "em": 0.25,
                "f1": 0.5,
                "rouge_l": 0.75,
                "first_token_em": 0.9,
                "accuracy": 0.8,
                "dataset/squad/em": 0.2,
                "dataset/squad/f1": 0.4,
                "dataset/squad/rouge_l": 0.6,
                "dataset/squad/count": 10,
                "dataset/foo/bar/em": 0.3,
                "dataset/foo/bar/f1": 0.5,
                "dataset/foo/bar/rouge_l": 0.7,
                "dataset/foo/bar/count": 2,
            }
        }
    )
    assert all(label in table for label in ("dataset", "EM", "F1", "ROUGE-L", "first_token_em", "accuracy", "count"))
    assert "| foo/bar " in table
    assert "| squad " in table
    assert "| overall " in table
    assert "0.9000" in table
    assert "0.8000" in table


def test_window_progress_total_counts_only_batches_intersecting_rank_rows():
    from src.evaluator import _window_batch_count, row_window

    assert _window_batch_count(row_window(1280, 0, 8), 24) == 7
    assert _window_batch_count(row_window(1280, 1, 8), 24) == 8
    assert _window_batch_count(row_window(1280, 7, 8), 24) == 8
    assert _window_batch_count((0, 160), None) is None


def test_evaluator_has_no_teacher_forced_path():
    from src.evaluator import Evaluator
    import inspect

    assert not hasattr(Evaluator, "teacher_forced")
    assert "include_teacher_metrics" not in inspect.signature(
        Evaluator.autoregressive
    ).parameters
