"""Run the repository's dataset evaluators serially and collect one manifest."""
from __future__ import annotations
import argparse, json, subprocess, sys
import hashlib
from datetime import datetime
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
        path.resolve() for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES
    )
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint files ({', '.join(sorted(CHECKPOINT_SUFFIXES))}) under {directory}")
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
    result = ["--machine", machine, "--datasets", plan["dataset"], "--split", plan["split"]]
    if plan["version"]:
        result += ["--source-version", plan["version"]]
    return result

def run(label, command, log_path, dry_run):
    print(f"[{label}] {' '.join(command)}", flush=True)
    if dry_run:
        return "planned"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    return "ok" if result.returncode == 0 else f"failed:{result.returncode}"

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", default="4090")
    parser.add_argument("--data-root", default="/data/lz/contexts/aggregated")
    parser.add_argument("--checkpoint", "--ckpt", dest="checkpoints", nargs="+")
    parser.add_argument("--checkpoint-dir", help="Recursively evaluate every .pt/.pth/.bin checkpoint")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--memory-bs", type=int, default=1)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output-dir", help="Exact directory for this run's manifest, logs and evaluation reports")
    output.add_argument("--output-root", default="outputs/all_suite", help="Parent for an automatically timestamped run directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.checkpoints) == bool(args.checkpoint_dir):
        parser.error("provide exactly one of --checkpoint or --checkpoint-dir")
    checkpoints = [Path(path).resolve() for path in args.checkpoints] if args.checkpoints else discover_checkpoints(args.checkpoint_dir)
    checkpoints = list(dict.fromkeys(checkpoints))
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            parser.error(f"checkpoint does not exist: {checkpoint}")
    if args.memory_bs < 1:
        parser.error("--memory-bs must be positive")
    root = Path(args.data_root)
    out = Path(args.output_dir) if args.output_dir else Path(args.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for plan in plans(root, args.datasets):
        common = common_args(plan, args.machine)
        for checkpoint in checkpoints:
            label = f"memory:{checkpoint.stem}"
            command = [sys.executable, "-m", "utils.launcher", "test", "--ckpt", str(checkpoint), "--data-root", str(root.resolve()), "--bs", str(args.memory_bs), *common]
            identity = hashlib.sha256(str(checkpoint).encode()).hexdigest()[:12]
            report_dir = out / plan['name'] / f"{checkpoint.stem}-{identity}"
            command += ["--output-dir", str(report_dir)]
            log_name = f"{plan['name']}-{checkpoint.stem}-{identity}.log"
            status = run(label, command, out / log_name, args.dry_run)
            manifest.append({"dataset": plan["name"], "split": plan["split"], "model": label, "checkpoint": str(checkpoint), "output_dir": str(report_dir), "log": str(out / log_name), "status": status, "command": command})
            (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(out), "runs": len(manifest), "failed": sum(x["status"].startswith("failed") for x in manifest)}, ensure_ascii=False))
    if any(x["status"].startswith("failed") for x in manifest):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
