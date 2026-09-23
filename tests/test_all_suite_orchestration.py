from __future__ import annotations

"""Regression tests for the multi-checkpoint evaluation orchestrator."""

import importlib.util
import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test_all_suite_script", ROOT / "scripts/evaluation/test_all_suite.py"
)
test_all_suite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(test_all_suite)


def _dataset(root: Path, name: str = "squad") -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "validation.jsonl").write_text("{}\n")


def test_checkpoint_stem_names_output_directory_without_manifest(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoints" / "base.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    data_root = tmp_path / "data"
    _dataset(data_root)
    output_root = tmp_path / "evaluation"

    commands = []

    def record_run(label, command, dry_run):
        commands.append(command)
        return "planned"

    monkeypatch.setattr(test_all_suite, "run", record_run)
    test_all_suite.main(
        [
            "--ckpt",
            str(checkpoint),
            "--data-root",
            str(data_root),
            "--datasets",
            "squad",
            "--output-dir",
            str(output_root),
            "--dry-run",
        ]
    )

    checkpoint_output = output_root / "base"
    assert checkpoint_output.is_dir()
    assert not (checkpoint_output / "manifest.json").exists()
    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--output-dir") + 1] == str(checkpoint_output)
    assert command[command.index("--report-file") + 1] == str(checkpoint_output / "squad.json")
    assert "--teacher-forced" not in command


def test_report_filename_removes_dataset_separators():
    assert test_all_suite.report_filename("squad") == "squad.json"
    assert test_all_suite.report_filename("ms_marco_v1") == "msmarcov1.json"


def test_duplicate_checkpoint_stems_cannot_overwrite_each_other(tmp_path):
    checkpoints = []
    for parent in ("first", "second"):
        checkpoint = tmp_path / parent / "base.pt"
        checkpoint.parent.mkdir()
        checkpoint.touch()
        checkpoints.append(checkpoint)
    data_root = tmp_path / "data"
    _dataset(data_root)

    with pytest.raises(SystemExit):
        test_all_suite.main(
            [
                "--ckpt",
                *(str(path) for path in checkpoints),
                "--data-root",
                str(data_root),
                "--datasets",
                "squad",
                "--output-dir",
                str(tmp_path / "evaluation"),
                "--dry-run",
            ]
        )


def test_summary_uses_dataset_order_and_percentage_with_two_decimals(tmp_path):
    checkpoints = [tmp_path / "base.pt", tmp_path / "off.pt"]
    datasets = ["squad", "hotpotqa"]
    reports = {}
    values = {
        ("base", "squad"): (0.4973501, 0.6382467, 0.6362951),
        ("base", "hotpotqa"): (0.1, 0.2, 0.3),
        ("off", "squad"): (0.01234, 0.0, 1.0),
    }
    for checkpoint in checkpoints:
        for dataset in datasets:
            metrics = values.get((checkpoint.stem, dataset))
            if metrics is None:
                continue
            report = tmp_path / checkpoint.stem / dataset / "report.json"
            report.parent.mkdir(parents=True)
            report.write_text(
                json.dumps(
                    {
                        "autoregressive": dict(zip(("em", "f1", "rouge_l"), metrics)),
                        "teacher_forced": None,
                    }
                )
            )
            reports[(checkpoint, dataset)] = report

    summary = tmp_path / "summary.csv"
    test_all_suite.write_summary(summary, checkpoints, datasets, reports)

    with summary.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == [
        "ckpt",
        "squad_AR_EM",
        "squad_AR_F1",
        "squad_AR_ROUGE-L",
        "hotpotqa_AR_EM",
        "hotpotqa_AR_F1",
        "hotpotqa_AR_ROUGE-L",
    ]
    assert rows[1] == ["base", "49.74", "63.82", "63.63", "10.00", "20.00", "30.00"]
    assert rows[2] == ["off", "1.23", "0.00", "100.00", "", "", ""]
