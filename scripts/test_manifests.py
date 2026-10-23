"""Evaluate a MemoryQwen checkpoint on the shared ICL sample manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table


DEFAULT_MANIFESTS = [
    "/data/lz/lora_ablation/icl_eval/manifests/qwen3_test_seed42.json",
    "/data/lz/lora_ablation/icl_eval/manifests/qwen3_test_seed27.json",
    "/data/lz/lora_ablation/icl_eval/manifests/qwen3_test_seed72.json",
]


def manifest_rows(manifest: dict) -> list[dict]:
    rows = []
    for dataset, data in manifest["datasets"].items():
        for context in data["contexts"]:
            pairs = []
            for question in context["questions"]:
                references = [str(value).strip() for value in question["references"] if str(value).strip()]
                if not references:
                    continue
                pairs.append({
                    "question": question["question"],
                    "answer": references[0],
                    "answers": references,
                    "metadata": {"options": question.get("options", [])},
                })
            if pairs:
                rows.append({
                    "context": context["context"],
                    "dataset": dataset,
                    "context_id": context.get("context_id", ""),
                    "qa_pairs": pairs,
                })
    return rows


def materialize_manifest(path: Path, output_dir: Path) -> Path:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = manifest_rows(manifest)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    target = output_dir / f"{path.stem}_{digest}.jsonl"
    if not target.exists():
        target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return target


def print_summary(report: dict, manifest: Path, console: Console) -> None:
    metrics = report.get("qa_task", {})
    datasets = sorted({
        key.removeprefix("dataset/").rsplit("/", 1)[0]
        for key in metrics
        if key.startswith("dataset/") and key.endswith("/count")
    })
    table = Table(
        title=f"Manifest summary · {manifest.name}",
        box=box.Box(" ━━ \n" "    \n" " ━━ \n" "    \n" "    \n" "    \n" "    \n" " ━━ \n"),
        header_style="bold cyan",
    )
    for column in ("dataset", "EM", "F1", "ROUGE-L", "QA"):
        table.add_column(column, justify="left" if column == "dataset" else "right")
    for dataset in [*datasets, "overall"]:
        prefix = "" if dataset == "overall" else f"dataset/{dataset}/"
        count = sum(metrics.get(f"dataset/{name}/count", 0) for name in datasets) if dataset == "overall" else metrics.get(f"{prefix}count", 0)
        table.add_row(
            dataset,
            f"{metrics.get(f'{prefix}em', 0.0):.4f}",
            f"{metrics.get(f'{prefix}f1', 0.0):.4f}",
            f"{metrics.get(f'{prefix}rouge_l', 0.0):.4f}",
            f"{count:,.0f}",
        )
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifests", nargs="+", default=DEFAULT_MANIFESTS)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--context-batch-size", "--batch-size", type=int, default=8)
    parser.add_argument("--qa-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3")
    args = parser.parse_args()
    if args.context_batch_size < 1 or args.qa_batch_size < 1 or args.max_new_tokens < 1:
        parser.error("batch sizes and max-new-tokens must be >= 1")
    checkpoint = Path(args.checkpoint).resolve()
    root = Path(args.output_dir).resolve() if args.output_dir else checkpoint.parent / "manifest_eval"
    root.mkdir(parents=True, exist_ok=True)
    input_dir = root / "inputs"
    input_dir.mkdir(exist_ok=True)
    console = Console()
    torchrun = shutil.which("torchrun")
    if torchrun is None:
        raise RuntimeError("torchrun is required; activate the MemoryQwen conda environment")

    for manifest_arg in args.manifests:
        manifest_path = Path(manifest_arg).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        data_file = materialize_manifest(manifest_path, input_dir)
        seed = int(manifest["seed"])
        report_file = root / f"{manifest_path.stem}.report.json"
        console.print(f"[bold]Evaluating seed={seed}[/bold] ({manifest_path})")
        command = [
            torchrun, "--standalone", f"--nproc-per-node={args.gpus}",
            "-m", "scripts.test",
            "--checkpoint", str(checkpoint), "--split", "test",
            "--data-file", str(data_file), "--max-context-tokens", str(manifest["max_context_tokens"]),
            "--max-contexts-per-dataset", str(manifest["max_contexts"]),
            "--batch-size", str(args.context_batch_size), "--qa-batch-size", str(args.qa_batch_size),
            "--max-new-tokens", str(args.max_new_tokens), "--report-file", str(report_file),
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        subprocess.run(command, cwd=Path(__file__).resolve().parent.parent, env=env, check=True)
        print_summary(json.loads(report_file.read_text(encoding="utf-8")), manifest_path, console)


if __name__ == "__main__":
    main()
