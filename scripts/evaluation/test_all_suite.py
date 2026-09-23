"""Run dataset evaluators serially and write compact reports plus one summary."""

from __future__ import annotations
import argparse, csv, json, subprocess, sys
from pathlib import Path

DATASETS = ("squad", "ms_marco_v1", "ms_marco_v2", "hotpotqa", "race")
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".bin"}


def _nonempty(path):
    return path.is_file() and path.stat().st_size > 0


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


def plans(root, selected):
    for name in selected:
        if name == "ms_marco_v1":
            dataset, version = "ms_marco", "ms_marco_v1_1"
            split = "test" if _nonempty(root / dataset / "test.jsonl") else "validation"
        elif name == "ms_marco_v2":
            dataset, version, split = "ms_marco", "ms_marco_v2_1", "validation"
        else:
            dataset, version = name, None
            split = "test" if _nonempty(root / dataset / "test.jsonl") else "validation"
        path = root / dataset / f"{split}.jsonl"
        if not _nonempty(path):
            raise FileNotFoundError(f"no nonempty {split}.jsonl for {name}: {path}")
        yield {"name": name, "dataset": dataset, "version": version, "split": split}


def common_args(plan, machine):
    result = [
        "--machine",
        machine,
        "--datasets",
        plan["dataset"],
        "--split",
        plan["split"],
    ]
    if plan["version"]:
        result += ["--source-version", plan["version"]]
    return result


def run(label, command, dry_run):
    print(f"[{label}] {' '.join(command)}", flush=True)
    if dry_run:
        return "planned"
    # Inherit the terminal directly so tqdm remains interactive without retaining log files.
    result = subprocess.run(command)
    return "ok" if result.returncode == 0 else f"failed:{result.returncode}"


def write_summary(path, checkpoints, datasets, reports):
    metrics = ("em", "f1", "rouge_l")
    header = ["ckpt"]
    for dataset in datasets:
        header.extend(
            (f"{dataset}_AR_EM", f"{dataset}_AR_F1", f"{dataset}_AR_ROUGE-L")
        )
    rows = []
    for checkpoint in checkpoints:
        row = [checkpoint.stem]
        for dataset in datasets:
            report_path = reports.get((checkpoint, dataset))
            values = None
            if report_path is not None and report_path.is_file():
                report = json.loads(report_path.read_text())
                values = report.get("autoregressive")
            for metric in metrics:
                value = None if values is None else values.get(metric)
                row.append("" if value is None else f"{100 * float(value):.2f}")
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    return rows


def report_filename(dataset):
    """Return the compact report filename used inside a checkpoint directory."""
    return f"{dataset.replace('_', '')}.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", default="4090")
    parser.add_argument("--data-root", default="/data/lz/contexts/aggregated")
    parser.add_argument("--checkpoint", "--ckpt", dest="checkpoints", nargs="+")
    parser.add_argument(
        "--ckpt-dir", help="Recursively evaluate every .pt/.pth/.bin checkpoint"
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument("--memory-bs", type=int, default=16)
    parser.add_argument("--max-context-tokens", type=int, default=1024)
    parser.add_argument("--max-context-rows", type=int, default=5000)
    parser.add_argument("--qa-per-context", type=int, default=8)
    parser.add_argument("--context-shuffle-seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="outputs/eval_ab_reader",
        help="Root directory; each checkpoint is written under <output-dir>/<checkpoint-stem>",
    )
    parser.add_argument(
        "--teacher-forced",
        action="store_true",
        help="Also run teacher-forced diagnostics for every checkpoint (disabled by default)",
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
    root = Path(args.data_root)
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
        for plan in plans(root, args.datasets):
            common = common_args(plan, args.machine)
            label = f"memory:{checkpoint.stem}"
            command = [
                sys.executable,
                "-m",
                "utils.launcher",
                "test",
                "--ckpt",
                str(checkpoint),
                "--data-root",
                str(root.resolve()),
                "--bs",
                str(args.memory_bs),
                "--max-context-tokens",
                str(args.max_context_tokens),
                *common,
            ]
            if args.max_context_rows is not None:
                command += ["--max-context-rows", str(args.max_context_rows)]
            if args.qa_per_context is not None:
                command += ["--qa-per-context", str(args.qa_per_context)]
            command += ["--context-shuffle-seed", str(args.context_shuffle_seed)]
            report_path = checkpoint_out / report_filename(plan["name"])
            command += [
                "--output-dir",
                str(checkpoint_out),
                "--report-file",
                str(report_path),
            ]
            if args.teacher_forced:
                command += ["--teacher-forced"]
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
