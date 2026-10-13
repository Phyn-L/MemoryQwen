#!/usr/bin/env python3
"""Build clean full train and validation C1QA JSONL files."""

from __future__ import annotations

import json
from pathlib import Path

from split_c1qa_eval_sets import cache_rows, make_row


ROOT = Path("data/generated")
ALL = ROOT / "1qa-all.jsonl"
DATASETS = (
    "msmarcov1.1",
    "msmarcov2.1",
    "squad",
    "race",
    "duorc/selfrc",
    "duorc/paraphraserc",
    "hotpotqa(distractor)",
    "musique",
    "2wikimultihopqa",
)
VALIDATION_SPLITS = {
    "msmarcov1.1": "validation",
    "msmarcov2.1": "validation",
    "squad": "validation",
    "race": "validation",
    "duorc/selfrc": "validation",
    "duorc/paraphraserc": "validation",
    "hotpotqa(distractor)": "validation",
    "musique": "dev",
    "2wikimultihopqa": "validation",
}


def read_existing() -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    with ALL.open() as fh:
        for line in fh:
            row = json.loads(line)
            grouped.setdefault((row["dataset"], row["source_split"]), []).append(row)
    return grouped


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    if not ALL.exists():
        raise FileNotFoundError(ALL)
    existing = read_existing()

    train_rows: list[dict] = []
    with ALL.open() as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("source_split") == "train":
                train_rows.append(row)

    validation_rows: list[dict] = []
    manifest: dict[str, dict] = {}
    for dataset in DATASETS:
        target = VALIDATION_SPLITS[dataset]
        rows = existing.get((dataset, target), [])
        source = "1qa-all.jsonl"
        if not rows:
            raw = cache_rows(dataset, target)
            rows = [
                converted
                for i, raw_row in enumerate(raw)
                if (converted := make_row(dataset, target, raw_row, i)) is not None
            ]
            source = "hf_cache"
        if not rows:
            raise RuntimeError(f"No validation rows for {dataset} ({target})")
        validation_rows.extend(rows)
        manifest[dataset] = {
            "source_split": target,
            "rows": len(rows),
            "source": source,
        }

    write_jsonl(ROOT / "1qa-train.jsonl", train_rows)
    write_jsonl(ROOT / "1qa-validation.jsonl", validation_rows)
    manifest["train"] = {"source_split": "train", "rows": len(train_rows)}
    manifest["validation_total"] = len(validation_rows)
    (ROOT / "1qa-clean-splits.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
