"""Run the memory-side evaluator over checkpoints x datasets and write one summary.

For every checkpoint and every dataset this orchestrator runs ``scripts/test.py`` on that
dataset's own file (``src.evaluation.eval_data`` decides where the file lives) and collects the
per-dataset report into a single ``summary.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from src.evaluation.eval_data import DEFAULT_DATASETS, report_name, resolve, slug
from utils.machines import machine_names
from utils.model_paths import machine_paths

DATASETS = DEFAULT_DATASETS
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".bin"}


def discover_checkpoints(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {directory}")
    checkpoints = sorted(
        path.resolve()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES
    )
    if not checkpoints:
        raise FileNotFoundError(
            f"no checkpoint files ({', '.join(sorted(CHECKPOINT_SUFFIXES))}) under {directory}"
        )
    return checkpoints


def plans(root, selected, split=None):
    """One plan per dataset: the file to evaluate plus the split it came from."""
    for name in selected:
        path, actual_split = resolve(root, name, split)
        yield {"name": name, "path": str(path), "split": actual_split}


def common_args(plan):
    """The dataset-specific arguments of one evaluator run.

    ``--data-file`` is what makes a per-dataset run possible: the checkpoint's config names one
    split file for the whole corpus, and the evaluator is pointed at this dataset's file
    instead. Every flag here must exist in ``scripts/test.py``.
    """
    return ["--split", plan["split"], "--data-file", plan["path"]]


def run(label, command, dry_run):
    print(f"[{label}] {' '.join(command)}", flush=True)
    if dry_run:
        return "planned"
    # Inherit the terminal directly so tqdm remains interactive without retaining log files.
    result = subprocess.run(command)
    return "ok" if result.returncode == 0 else f"failed:{result.returncode}"


def write_summary(path, checkpoints, datasets, reports):
    """CSV of the answer metrics plus the teacher-forced reconstruction score per dataset."""
    header = ["ckpt"]
    for dataset in datasets:
        name = slug(dataset)
        header.extend(
            f"{name}_AR_{label}" for label in ("EM", "F1", "ROUGE-L", "recon_CE")
        )
        if dataset == "race":
            header.append(f"{name}_AR_Acc")
    rows = []
    for checkpoint in checkpoints:
        row = [checkpoint.stem]
        for dataset in datasets:
            report_path = reports.get((checkpoint, dataset))
            answers, recon = None, {}
            if report_path is not None and report_path.is_file():
                report = json.loads(report_path.read_text())
                # The evaluator reports one group per task: generated QA answers
                # (``qa_task``) and the teacher-forced reconstruction (``recon_task``).
                answers = report.get("qa_task")
                recon = report.get("recon_task") or {}
            for metric in ("em", "f1", "rouge_l"):
                value = None if answers is None else answers.get(metric)
                row.append("" if value is None else f"{100 * float(value):.2f}")
            recon_ce = None if answers is None else recon.get("recon_ce")
            row.append("" if recon_ce is None else f"{float(recon_ce):.4f}")
            if dataset == "race":
                value = None if answers is None else answers.get("accuracy")
                row.append("" if value is None else f"{100 * float(value):.2f}")
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", choices=machine_names(), default=None)
    parser.add_argument(
        "--data-root",
        help="Data tree written by scripts/data/ (defaults to the DATA_ROOT for --machine)",
    )
    parser.add_argument("--checkpoint", "--ckpt", dest="checkpoints", nargs="+")
    parser.add_argument(
        "--ckpt-dir", help="Recursively evaluate every .pt/.pth/.bin checkpoint"
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument(
        "--split",
        choices=("test", "validation"),
        help="Evaluation split; default picks the fixed test split, else the sampled subset",
    )
    parser.add_argument("--memory-bs", type=int, default=16)
    parser.add_argument("--max-context-tokens", type=int, default=2048)
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Cap on the contexts of each dataset (scripts/test.py --max-samples)",
    )
    parser.add_argument("--max-new-tokens", type=int, help="Decoding budget per answer")
    parser.add_argument("--qa-batch-size", type=int, help="QA rows per generation group")
    parser.add_argument(
        "--output-dir",
        default="outputs/eval_ab_reader",
        help="Root directory; each checkpoint is written under <output-dir>/<checkpoint-stem>",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.checkpoints) == bool(args.ckpt_dir):
        parser.error("provide exactly one of --checkpoint/--ckpt or --ckpt-dir")
    checkpoints = (
        [Path(path).resolve() for path in args.checkpoints]
        if args.checkpoints
        else discover_checkpoints(args.ckpt_dir)
    )
    checkpoints = list(dict.fromkeys(checkpoints))
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            parser.error(f"checkpoint does not exist: {checkpoint}")
    if args.memory_bs < 1:
        parser.error("--memory-bs must be positive")
    _, paths = machine_paths(args.machine)
    root = Path(args.data_root or paths["DATA_ROOT"])
    output_root = Path(args.output_dir)
    output_dirs = {}
    for checkpoint in checkpoints:
        if checkpoint.stem in output_dirs:
            parser.error(
                "checkpoint names must be unique because output directories use their stems: "
                f"{output_dirs[checkpoint.stem]} and {checkpoint}"
            )
        output_dirs[checkpoint.stem] = checkpoint
    output_root.mkdir(parents=True, exist_ok=True)
    total_runs = failed_runs = 0
    reports = {}
    for checkpoint in checkpoints:
        checkpoint_out = output_root / checkpoint.stem
        checkpoint_out.mkdir(parents=True, exist_ok=True)
        if not args.dry_run:
            (checkpoint_out / "manifest.json").unlink(missing_ok=True)
        for plan in plans(root, args.datasets, args.split):
            label = f"memory:{checkpoint.stem}"
            report_path = checkpoint_out / report_name(plan["name"])
            command = [
                sys.executable,
                "-m",
                "utils.launcher",
                "test",
                "--ckpt",
                str(checkpoint),
                "--bs",
                str(args.memory_bs),
                "--max-context-tokens",
                str(args.max_context_tokens),
                *common_args(plan),
                "--output-dir",
                str(checkpoint_out),
                "--report-file",
                str(report_path),
            ]
            if args.machine:
                command += ["--machine", args.machine]
            if args.max_samples is not None:
                command += ["--max-samples", str(args.max_samples)]
            if args.max_new_tokens is not None:
                command += ["--max-new-tokens", str(args.max_new_tokens)]
            if args.qa_batch_size is not None:
                command += ["--qa-batch-size", str(args.qa_batch_size)]
            if not args.dry_run:
                report_path.unlink(missing_ok=True)
            status = run(label, command, args.dry_run)
            if status == "ok":
                reports[(checkpoint, plan["name"])] = report_path
            total_runs += 1
            failed_runs += status.startswith("failed")
    summary_path = output_root / "summary.csv"
    if not args.dry_run:
        write_summary(summary_path, checkpoints, args.datasets, reports)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "checkpoint_outputs": [
                    str(output_root / checkpoint.stem) for checkpoint in checkpoints
                ],
                "runs": total_runs,
                "failed": failed_runs,
                "summary": None if args.dry_run else str(summary_path),
            },
            ensure_ascii=False,
        )
    )
    if failed_runs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
