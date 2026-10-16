"""Evaluation logging contract.

Two passes report into SwanLab: the autoregressive QA pass (answer metrics) and the
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
spec = importlib.util.spec_from_file_location(
    "_train_script", REPO / "scripts/train.py"
)
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

ANSWER_METRIC_KEYS = ("em", "f1", "rouge_l")


def test_generated_answer_metrics_and_dataset_metrics_are_logged():
    metrics = {key: 0.7 for key in ANSWER_METRIC_KEYS}
    metrics.update(
        {
            "dataset/squad/em": 0.8,
            "dataset/squad/f1": 0.7,
            "dataset/squad/rouge_l": 0.6,
            "dataset/squad/count": 256,
            "dataset/hotpotqa(distractor)/f1": 0.5,
        }
    )
    payloads = dict(train.eval_log_payloads({"qa_task": metrics, "recon_task": metrics}))
    for section in ("eval_qa", "eval_recon"):
        payload = payloads[section]
        assert payload["squad_EM"] == 0.8
        assert payload["squad_F1"] == 0.7
        assert payload["squad_ROUGE-L"] == 0.6
        assert "squad_count" not in payload
        assert payload["hotpotqa_distractor_F1"] == 0.5
        for key in ANSWER_METRIC_KEYS:
            assert payload[key] == 0.7
        assert "first_token_em" not in payload
    assert set(payloads) == {"eval_qa", "eval_recon"}
    assert all(
        not key.startswith(("eval_qa/", "eval_recon/"))
        for payload in payloads.values()
        for key in payload
    )


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
    (section, payload), = train.eval_log_payloads({"recon_task": metrics})
    assert section == "eval_recon"
    assert payload["recon_ce"] == 1.25
    assert payload["recon_ppl"] == 3.49
    assert payload["recon_token_accuracy"] == 0.42
    assert payload["contexts"] == 512
    assert payload["squad_recon_ce"] == 1.1
    assert payload["squad_contexts"] == 256
    assert payload["hotpotqa_distractor_recon_ce"] == 1.4


def test_evaluation_payloads_cover_exactly_what_was_returned():
    assert train.eval_log_payloads() == []
    values = {"f1": 0.5, "recon_token_accuracy": 0.9, "dataset/squad/count": 3}
    assert train.eval_log_payloads({"qa_task": values}) == [
        ("eval_qa", {"f1": 0.5, "recon_token_accuracy": 0.9})
    ]
    # Non-scalars and tasks that returned nothing produce no payload.
    assert train.eval_log_payloads({"qa_task": {"records": [1, 2]}}) == []


def test_autoregressive_reconstruction_uses_the_reconstruction_section():
    assert train.eval_log_payloads(
        {"recon_autoregressive": {"recon_token_accuracy": 0.75}}
    ) == [("eval_recon", {"recon_token_accuracy": 0.75})]


def test_eval_metrics_table_groups_datasets_and_metrics():
    table = train.format_eval_metrics_table(
        {
            "qa_task": {
                "em": 0.25,
                "f1": 0.5,
                "rouge_l": 0.75,
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
    assert all(
        label in table
        for label in (
            "dataset",
            "EM",
            "F1",
            "ROUGE-L",
            "contexts",
            "QA",
            "Recon contexts",
        )
    )
    assert "foo/bar" in table
    assert "squad" in table
    assert "overall" in table
    assert "accuracy" not in table
    assert "first_token_em" not in table


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
    assert all(
        label in table
        for label in ("recon_ce", "recon_ppl", "token_acc", "contexts")
    )
    assert "squad" in table
    assert "overall" in table
    assert "1.2500" in table
    assert "0.4200" in table


def test_eval_report_keeps_global_counts_and_wraps_metadata_to_table_width():
    from types import SimpleNamespace

    selection = SimpleNamespace(
        contexts=tuple(range(5120)),
        qa_rows=11635,
        qa_counts=(("squad", 4933),),
        context_counts=(("squad", 1024),),
    )
    table = train.format_eval_metrics_table(
        {"qa_task": {"dataset/squad/count": 4933}},
        selection=selection,
        seed=42,
        step=2500,
        max_contexts=1024,
        max_new_tokens=64,
    )
    assert all(value in table for value in ("1,024", "4,933", "5,120", "11,635"))
    assert "Step 2,500" in table
    assert "global sample" in table and "seed 42" in table
    assert "sampled counts" not in table
    assert sum("━━━" in line for line in table.splitlines()) == 3
    assert len(table.splitlines()[0]) < 120


def test_window_progress_total_counts_only_batches_intersecting_rank_rows():
    from src.evaluator import _window_batch_count, row_window

    assert _window_batch_count(row_window(1280, 0, 8), 24) == 7
    assert _window_batch_count(row_window(1280, 1, 8), 24) == 8
    assert _window_batch_count(row_window(1280, 7, 8), 24) == 8
    assert _window_batch_count((0, 160), None) is None


def test_eval_progress_only_main_rank_writes_to_terminal(monkeypatch):
    import io

    from rich.console import Console

    from src import evaluator

    for rank, enabled in ((0, True), (1, True), (1, False), (0, False)):
        output = io.StringIO()
        monkeypatch.setattr(
            evaluator,
            "Console",
            lambda **kwargs: Console(file=output, force_terminal=True, width=120),
        )
        progress, task = evaluator._make_eval_progress(
            2, "Validation", rank, enabled, "qa_rows"
        )
        try:
            progress.update(task, advance=1, refresh=True)
        finally:
            progress.stop()
        assert bool(output.getvalue()) == (rank == 0 and enabled)


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
    assert clock.recon_teacherforce is True
    assert clock.recon_autoregressive is False
