#!/usr/bin/env python3
"""Plot Qwen3 token-length distributions for the generated C1QA dataset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from transformers import AutoTokenizer


FIELDS = ("context", "question", "answer")
COLORS = {"context": "#2563eb", "question": "#16a34a", "answer": "#dc2626"}
PLOT_MAX = {"context": 4096, "question": 256, "answer": 256}


def _find_tokenizer(path: Path) -> Path:
    if (path / "tokenizer.json").exists():
        return path
    snapshots = sorted(path.glob("snapshots/*"))
    if snapshots:
        return snapshots[-1]
    raise FileNotFoundError(f"Qwen tokenizer not found under {path}")


def _lengths(tokenizer, texts: list[str], batch_size: int) -> np.ndarray:
    values: list[int] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        values.extend(min(len(ids), 4096) for ids in encoded["input_ids"])
    return np.asarray(values, dtype=np.int32)


def _plot_one(dataset: str, stats: dict[str, np.ndarray], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, field in zip(axes, FIELDS):
        values = stats[field]
        plot_max = PLOT_MAX[field]
        plot_values = np.minimum(values, plot_max)
        counts, edges = np.histogram(plot_values, bins=128, range=(0, plot_max), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        curve = gaussian_filter1d(counts, sigma=2.0)
        ax.fill_between(centers, counts, color=COLORS[field], alpha=0.20)
        ax.plot(centers, curve, color=COLORS[field], linewidth=2)
        ax.axvline(min(values.min(), plot_max), color="#111827", linestyle=":", linewidth=1.3)
        ax.axvline(min(values.max(), plot_max), color="#111827", linestyle="--", linewidth=1.3)
        ax.axvline(min(values.mean(), plot_max), color="#f59e0b", linestyle="-", linewidth=1.5)
        ax.axvline(min(np.median(values), plot_max), color="#7c3aed", linestyle="-.", linewidth=1.5)
        percentiles = {p: float(np.percentile(values, p)) for p in (80, 90, 95)}
        percentile_colors = {80: "#0891b2", 90: "#c026d3", 95: "#ea580c"}
        for percentile, value in percentiles.items():
            plot_value = min(value, plot_max)
            ax.axvline(
                plot_value,
                color=percentile_colors[percentile],
                linestyle=(0, (1, 1.5)),
                linewidth=1.4,
                label=f"p{percentile}={value:.0f}" if value <= plot_max else f"p{percentile}>={plot_max}",
            )
        ax.set_xlim(0, plot_max)
        ax.set_xlabel(f"Qwen3 tokens (display clipped at {plot_max})")
        ax.set_ylabel("Density")
        ax.set_title(
            f"{field}\n"
            f"min={values.min()}  max={values.max()}  "
            f"mean={values.mean():.1f}  median={np.median(values):.1f}\n"
            f"p80={percentiles[80]:.0f}  p90={percentiles[90]:.0f}  "
            f"p95={percentiles[95]:.0f}"
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax.grid(alpha=0.2)
    fig.suptitle(f"{dataset} token-length distributions", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/generated/ift_c1qa_from_hf_cache.jsonl"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("/data/lz/hf_cache/hub/models--Qwen--Qwen3-8B"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated/c1qa_token_distributions"))
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        _find_tokenizer(args.tokenizer), local_files_only=True, use_fast=True
    )
    texts: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    with args.input.open() as fh:
        for line in fh:
            row = json.loads(line)
            answer = row.get("answers", [""])[0] if row.get("answers") else ""
            texts[row["dataset"]]["context"].append(row.get("context", ""))
            texts[row["dataset"]]["question"].append(row.get("question", ""))
            texts[row["dataset"]]["answer"].append(answer)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_stats: dict[str, dict[str, dict[str, float | int]]] = {}
    for dataset, columns in texts.items():
        arrays = {field: _lengths(tokenizer, columns[field], args.batch_size) for field in FIELDS}
        _plot_one(dataset, arrays, args.output_dir / f"{dataset.replace('/', '_').replace('(', '_').replace(')', '')}.png")
        all_stats[dataset] = {
            field: {
                "min": int(values.min()),
                "max": int(values.max()),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "p80": float(np.percentile(values, 80)),
                "p90": float(np.percentile(values, 90)),
                "p95": float(np.percentile(values, 95)),
                "count": int(values.size),
                "clipped_at_4096": int(np.sum(values == 4096)),
            }
            for field, values in arrays.items()
        }
    (args.output_dir / "statistics.json").write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(all_stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
