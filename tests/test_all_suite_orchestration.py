"""Regression tests for the multi-checkpoint evaluation orchestrator.

It drives ``scripts/test.py`` once per (checkpoint, dataset); the dataset's file comes from
``src/eval_data.py``, which is what keeps this suite on the same data as the training pipeline.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test_all_suite_script", ROOT / "scripts/evaluation/test_all_suite.py"
)
test_all_suite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(test_all_suite)


def _dataset(root: Path, name: str = "squad", split: str = "test") -> Path:
    """Create one nonempty per-dataset evaluation file in the scripts/data layout."""
    directory = root / "1qa" / "meta" / split
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{test_all_suite.slug(name)}.json"
    path.write_text("{}\n")
    return path


def _checkpoint(tmp_path, name="base.pt"):
    checkpoint = tmp_path / "checkpoints" / name
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.touch()
    return checkpoint


@pytest.mark.parametrize("split", [None, "test"])
def test_checkpoint_stem_names_output_directory_without_manifest(tmp_path, monkeypatch, split):
    checkpoint = _checkpoint(tmp_path)
    data_root = tmp_path / "data"
    expected_file = _dataset(data_root)
    output_root = tmp_path / "evaluation"

    commands = []

    def record_run(label, command, dry_run):
        commands.append(command)
        return "planned"

    monkeypatch.setattr(test_all_suite, "run", record_run)
    argv = [
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
    if split:
        argv += ["--split", split]
    test_all_suite.main(argv)

    checkpoint_output = output_root / "base"
    assert checkpoint_output.is_dir()
    assert not (checkpoint_output / "manifest.json").exists()
    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--output-dir") + 1] == str(checkpoint_output)
    assert command[command.index("--report-file") + 1] == str(checkpoint_output / "squad.json")
    # The evaluator is pointed at this dataset's file, and only at flags scripts/test.py has.
    assert command[command.index("--data-file") + 1] == str(expected_file)
    assert command[command.index("--split") + 1] == "test"
    for unsupported in ("--system-prompt-mode", "--qa-per-context", "--source-version"):
        assert unsupported not in command


def test_plans_prefer_the_meta_file(tmp_path):
    data_root = tmp_path / "data"
    (data_root / "1qa" / "meta" / "validation").mkdir(parents=True)
    (data_root / "1qa" / "meta" / "validation" / "squad.json").write_text("{}\n")
    plan, = list(test_all_suite.plans(data_root, ["squad"]))
    assert plan["split"] == "validation"
    assert Path(plan["path"]).name == "squad.json"



def test_plans_report_the_missing_file_instead_of_a_random_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_1qa"):
        list(test_all_suite.plans(tmp_path / "empty", ["squad"]))


def test_report_filename_uses_the_dataset_slug():
    assert test_all_suite.report_name("squad") == "squad.json"
    assert test_all_suite.report_name("hotpotqa(distractor)") == "hotpotqa_distractor.json"
    assert test_all_suite.report_name("msmarcov1.1") == "msmarcov1.1.json"


def test_duplicate_checkpoint_stems_cannot_overwrite_each_other(tmp_path):
    checkpoints = []
    for parent in ("first", "second"):
        checkpoints.append(_checkpoint(tmp_path / parent))
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


def test_summary_reads_the_task_groups_and_keeps_dataset_order(tmp_path):
    checkpoints = [tmp_path / "base.pt", tmp_path / "off.pt"]
    datasets = ["squad", "race", "hotpotqa(distractor)"]
    reports = {}
    values = {
        ("base", "squad"): (0.4973501, 0.6382467, 0.6362951),
        ("base", "race"): (0.1, 0.2, 0.3, 0.75),
        ("base", "hotpotqa(distractor)"): (0.1, 0.2, 0.3),
        ("off", "squad"): (0.01234, 0.0, 1.0),
    }
    for checkpoint in checkpoints:
        for dataset in datasets:
            metrics = values.get((checkpoint.stem, dataset))
            if metrics is None:
                continue
            report = tmp_path / checkpoint.stem / test_all_suite.slug(dataset) / "report.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(
                json.dumps(
                    {
                        # The evaluator reports one group per task; the summary is the QA
                        # comparison plus the teacher-forced reconstruction score.
                        # Non-race fixtures deliberately carry no accuracy value:
                        # strict=False keeps the partial dict instead of raising.
                        "qa_task": dict(
                            zip(("em", "f1", "rouge_l", "accuracy"), metrics, strict=False)
                        ),
                        "recon_task": {"recon_ce": 1.5, "recon_token_accuracy": 0.4},
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
        "squad_AR_recon_CE",
        "race_AR_EM",
        "race_AR_F1",
        "race_AR_ROUGE-L",
        "race_AR_recon_CE",
        "race_AR_Acc",
        "hotpotqa_distractor_AR_EM",
        "hotpotqa_distractor_AR_F1",
        "hotpotqa_distractor_AR_ROUGE-L",
        "hotpotqa_distractor_AR_recon_CE",
    ]
    assert rows[1] == [
        "base", "49.74", "63.82", "63.63", "1.5000",
        "10.00", "20.00", "30.00", "1.5000", "75.00",
        "10.00", "20.00", "30.00", "1.5000",
    ]
    assert rows[2] == [
        "off", "1.23", "0.00", "100.00", "1.5000",
        "", "", "", "", "", "", "", "", "",
    ]


def test_summary_is_empty_where_a_report_is_missing(tmp_path):
    checkpoints = [tmp_path / "base.pt"]
    summary = tmp_path / "summary.csv"
    test_all_suite.write_summary(summary, checkpoints, ["squad"], {})
    with summary.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[1] == ["base", "", "", "", ""]
