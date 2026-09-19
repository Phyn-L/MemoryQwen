"""Sequential checkpoint + Qwen ICL evaluation; nonempty test takes precedence."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from utils.model_paths import machine_paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--machine", default="4090")
    p.add_argument("--ckpt", default="outputs/Qwen1.7B_20260917_213648.pt")
    p.add_argument("--memory-bs", type=int, default=1)
    p.add_argument("--icl-bs", type=int, default=2)
    p.add_argument("--output-root", default="outputs/all_suite")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    root = Path(machine_paths(args.machine)[1]["DATA_ROOT"])
    out = Path(args.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True)
    manifest = []
    plans = [
        ("squad", None, None),
        ("ms_marco", "ms_marco_v1_1", "test"),
        ("ms_marco", "ms_marco_v1_1", "validation"),
        ("ms_marco", "ms_marco_v2_1", "validation"),
        ("hotpotqa", None, None),
        ("race", None, None),
    ]
    for dataset, version, requested_split in plans:
        folder = root / dataset
        test = folder / "test.jsonl"
        split = requested_split or (
            "test" if test.exists() and test.stat().st_size else "validation"
        )
        path = folder / f"{split}.jsonl"
        if not path.exists() or not path.stat().st_size:
            raise FileNotFoundError(f"No nonempty test/validation: {folder}")
        name = version or dataset
        total = answered = 0
        for line in path.open():
            for qa in json.loads(line).get("qa_pairs", []):
                if version and qa.get("source_dataset") != version:
                    continue
                total += 1
                answered += bool(
                    qa.get("answers") and any(str(x).strip() for x in qa["answers"])
                )
        if not answered:
            manifest.append(
                dict(
                    dataset=name,
                    split=split,
                    total=total,
                    status="unscorable:no_reference_answers",
                )
            )
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(
                f"{name}: {total} rows, no reference answers; scoring skipped",
                flush=True,
            )
            continue
        common = ["--machine", args.machine, "--datasets", dataset, "--split", split]
        if version:
            common += ["--source-version", version]
        commands = [
            (
                "memory",
                [
                    sys.executable,
                    "-m",
                    "utils.launcher",
                    "test",
                    "--ckpt",
                    args.ckpt,
                    "--no-filter-long-context",
                    "--bs",
                    str(args.memory_bs),
                    *common,
                ],
            )
        ]
        for model in ("Qwen3-1.7B", "Qwen3-8B"):
            commands.append(
                (
                    model,
                    [
                        sys.executable,
                        "-m",
                        "utils.launcher",
                        "icl",
                        "--config",
                        "configs/icl_zeroshot.yaml",
                        "--model",
                        model,
                        "--bs",
                        str(args.icl_bs),
                        "--output-dir",
                        str(out / name / split / model),
                        *common,
                    ],
                )
            )
        for label, cmd in commands:
            print(dataset, split, label, " ".join(cmd), flush=True)
            status = "planned"
            if not args.dry_run:
                with (out / f"{name}-{split}-{label}.log").open("w") as log:
                    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
                status = (
                    "ok" if result.returncode == 0 else f"failed:{result.returncode}"
                )
            manifest.append(
                dict(
                    dataset=name,
                    total=total,
                    answered=answered,
                    split=split,
                    model=label,
                    status=status,
                    command=cmd,
                )
            )
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Reports and logs: {out}")
    if any(x["status"].startswith("failed") for x in manifest):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
