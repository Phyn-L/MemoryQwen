"""Evaluation logging contract.

Two passes report into Aim: the autoregressive QA pass (answer metrics) and the
teacher-forced reconstruction pass (``recon_*`` metrics). The payload builder is deliberately
generic -- it logs the scalars an evaluation returned, so a new metric only has to be returned
by the evaluator, not also registered here.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("_train_script", REPO / "scripts/train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

ANSWER_METRIC_KEYS = ("em", "f1", "rouge_l", "first_token_em", "accuracy")


def test_generated_answer_metrics_and_dataset_metrics_are_logged():
    metrics = {key: 0.7 for key in ANSWER_METRIC_KEYS}
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
        for key in ANSWER_METRIC_KEYS:
            assert payload[prefix + key] == 0.7
    assert all(key.startswith("val_autoregressive_") for key in payload)


def test_teacher_forced_reconstruction_metrics_are_logged():
    metrics = {
        "recon_ce": 1.25,
        "recon_ppl": 3.49,
        "recon_token_accuracy": 0.42,
        "contexts": 512,
        "dataset/squad/recon_ce": 1.1,
        "dataset/squad/recon_token_accuracy": 0.5,
        "dataset/squad/contexts": 256,
        "dataset/hotpotqa(distractor)/recon_ce": 1.4,
    }
    payload, = train.eval_log_payloads({"recon_task": metrics})
    prefix = "val_autoregressive_recon_task/"
    assert payload[prefix + "recon_ce"] == 1.25
    assert payload[prefix + "recon_ppl"] == 3.49
    assert payload[prefix + "recon_token_accuracy"] == 0.42
    assert payload[prefix + "contexts"] == 512
    assert payload[prefix + "squad_recon_ce"] == 1.1
    assert payload[prefix + "squad_contexts"] == 256
    assert payload[prefix + "hotpotqa_distractor_recon_ce"] == 1.4


def test_evaluation_payloads_cover_exactly_what_was_returned():
    assert train.eval_log_payloads() == []
    values = {"f1": 0.5, "recon_token_accuracy": 0.9, "dataset/squad/count": 3}
    assert train.eval_log_payloads({"qa_task": values}) == [
        {
            "val_autoregressive_qa_task/f1": 0.5,
            "val_autoregressive_qa_task/recon_token_accuracy": 0.9,
            "val_autoregressive_qa_task/squad_count": 3,
        }
    ]
    # Non-scalars and tasks that returned nothing produce no payload.
    assert train.eval_log_payloads({"qa_task": {"records": [1, 2]}}) == []


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


def test_eval_metrics_table_renders_the_reconstruction_pass():
    table = train.format_eval_metrics_table(
        {
            "recon_task": {
                "recon_ce": 1.25,
                "recon_ppl": 3.49,
                "recon_token_accuracy": 0.42,
                "contexts": 512,
                "dataset/squad/recon_ce": 1.0,
                "dataset/squad/recon_token_accuracy": 0.6,
                "dataset/squad/contexts": 256,
            }
        }
    )
    assert "teacher-forced" in table
    assert all(label in table for label in ("recon_ce", "recon_ppl", "token_acc", "contexts"))
    assert "| squad " in table
    assert "| overall " in table
    assert "1.2500" in table
    assert "0.4200" in table


def test_window_progress_total_counts_only_batches_intersecting_rank_rows():
    from src.evaluator import _window_batch_count, row_window

    assert _window_batch_count(row_window(1280, 0, 8), 24) == 7
    assert _window_batch_count(row_window(1280, 1, 8), 24) == 8
    assert _window_batch_count(row_window(1280, 7, 8), 24) == 8
    assert _window_batch_count((0, 160), None) is None


def test_the_evaluator_has_the_teacher_forced_reconstruction_path():
    """Recon is scored by one forward behind the memory, not by generating the context.

    The autoregressive Recon pass it replaces decoded ``max_context_tokens`` tokens per
    context, which is why the training loop used to skip Recon entirely.
    """
    from src.evaluator import Evaluator

    assert hasattr(Evaluator, "recon_teacher_forced")
    assert hasattr(Evaluator, "compare_reader_tasks")
    # The cheap metric decides by default; the expensive one is opt-in.
    from utils.config import EvaluationConfig

    clock = EvaluationConfig()
    assert clock.recon_metrics is True
    assert clock.recon_autoregressive is False
