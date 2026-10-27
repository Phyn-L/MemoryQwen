"""Autoregressively reconstruct manifest contexts with the memory and Recon embeddings."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

# Make direct execution (`python scripts/test_recon.py`) use the repository package imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.test import evaluation_config
from src.evaluation.eval_metrics import METRIC_KEYS, qa_metrics
from src.model import TASK_RECON, load_model
from utils.checkpoint import CheckpointManager
from utils.machines import machine_names


DEFAULT_MANIFEST = Path(
    "/data/lz/lora_ablation/icl_eval/manifests/qwen3_test_seed42.json"
)
DATASETS = ("squad", "msmarcov1.1")


def select_contexts(manifest: dict, per_dataset: int = 2) -> list[dict]:
    examples = []
    for dataset in DATASETS:
        contexts = manifest.get("datasets", {}).get(dataset, {}).get("contexts", [])
        if len(contexts) < per_dataset:
            raise ValueError(
                f"manifest has {len(contexts)} {dataset} contexts; need {per_dataset}"
            )
        for context in contexts[:per_dataset]:
            text = str(context.get("context", "")).strip()
            if not text:
                raise ValueError(f"manifest contains an empty {dataset} context")
            examples.append({
                "dataset": dataset,
                "context_id": str(context.get("context_id", "")),
                "context": text,
            })
    return examples


def build_report(examples: list[dict], references: list[str], predictions: list[str]) -> dict:
    if not (len(examples) == len(references) == len(predictions)):
        raise ValueError("examples, references and predictions must have matching lengths")
    rows = []
    for example, reference, prediction in zip(examples, references, predictions, strict=True):
        rows.append({
            **example,
            "reference_context": reference,
            "prediction": prediction,
            "metrics": qa_metrics(prediction, reference),
        })
    by_dataset = {}
    for dataset in dict.fromkeys(row["dataset"] for row in rows):
        selected = [row for row in rows if row["dataset"] == dataset]
        by_dataset[dataset] = {
            key: sum(row["metrics"][key] for row in selected) / max(1, len(selected))
            for key in METRIC_KEYS
        }
    return {
        "metrics": {
            "overall": {
                key: sum(row["metrics"][key] for row in rows) / max(1, len(rows))
                for key in METRIC_KEYS
            },
            "dataset": by_dataset,
        },
        "examples": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--context", help="Reconstruct this text instead of manifest samples.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--machine", choices=machine_names(), default=None)
    args = parser.parse_args()
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")

    manifest_path = None
    if args.context is not None:
        if not args.context.strip():
            parser.error("--context must not be empty")
        examples = [{"dataset": "custom", "context_id": "cli", "context": args.context.strip()}]
    else:
        manifest_path = args.manifest.expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        examples = select_contexts(manifest)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    cfg = evaluation_config(checkpoint, machine=args.machine)
    cfg.checkpoint.output_dir = str(checkpoint.parent)

    tokenizer, model = load_model(cfg)
    CheckpointManager(cfg.checkpoint.output_dir).load(checkpoint, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    encoded = tokenizer(
        [example["context"] for example in examples],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.data.max_context_tokens,
        add_special_tokens=False,
    )
    context_ids = encoded.input_ids.to(device)
    context_mask = encoded.attention_mask.to(device).bool()
    reference_context_tokens = int(context_mask.sum(dim=1).max().item())
    max_new_tokens = args.max_new_tokens or reference_context_tokens
    embed = model.qwen.get_input_embeddings()
    prefix = model.encode_context_prefix(embed(context_ids), context_mask)
    count = len(examples)
    generated = model.generate_answers_with_prefix(
        prefix,
        torch.arange(count, device=device),
        torch.empty((count, 0), dtype=torch.long, device=device),
        torch.empty((count, 0), dtype=torch.bool, device=device),
        tokenizer,
        max_new_tokens=max_new_tokens,
        group_by_context=False,
        task_id=TASK_RECON,
    )
    references = [
        tokenizer.decode(context_ids[i][context_mask[i]].tolist(), skip_special_tokens=True)
        for i in range(count)
    ]
    predictions = [
        tokenizer.decode(row.tolist(), skip_special_tokens=True).strip()
        for row in generated
    ]
    report = {
        "checkpoint": str(checkpoint),
        "manifest": str(manifest_path) if manifest_path is not None else None,
        "max_new_tokens": max_new_tokens,
        "prefix": "memory_embedding + recon_task_embedding",
        **build_report(examples, references, predictions),
    }
    output = args.output.expanduser().resolve() if args.output else checkpoint.parent / "recon_test_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    table = Table(title="Autoregressive context reconstruction")
    table.add_column("dataset")
    for metric in METRIC_KEYS:
        table.add_column(metric.upper(), justify="right")
    for dataset in (*report["metrics"]["dataset"], "overall"):
        metrics = report["metrics"]["overall"] if dataset == "overall" else report["metrics"]["dataset"][dataset]
        table.add_row(dataset, *(f"{metrics[key]:.4f}" for key in METRIC_KEYS))
    console = Console()
    console.print(table)
    console.print(f"Comparison saved: {output}")


if __name__ == "__main__":
    main()
