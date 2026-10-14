#!/usr/bin/env python3
"""Build clean full train and validation C1QA JSONL files."""

from __future__ import annotations

import json
from pathlib import Path

from split_1qa_eval_sets import cache_rows, make_row


ROOT = Path("contexts/1qa")
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


def aggregate_context_rows(rows: list[dict]) -> list[dict]:
    """Merge QA rows with the same dataset/split/context into one context row."""
    grouped: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        key = (
            str(row.get("dataset", "")),
            str(row.get("source_split", "")),
            str(row.get("context", "")),
        )
        current = grouped.get(key)
        if current is None:
            current = dict(row)
            current["qa_pairs"] = []
            grouped[key] = current
        pairs = row.get("qa_pairs")
        if not isinstance(pairs, list):
            conversation = row.get("conversations") or []
            answer = next(
                (
                    item.get("content", "")
                    for item in conversation
                    if item.get("role") == "assistant"
                ),
                "",
            )
            pairs = [{
                "id": row.get("id", ""),
                "question": row.get("question", ""),
                "answers": row.get("answers") or [],
                "answer": answer,
                "metadata": row.get("metadata") or {},
            }]
        for pair in pairs:
            current["qa_pairs"].append(pair)
    for row in grouped.values():
        row["id"] = row["qa_pairs"][0].get("id", "")
        row.pop("question", None)
        row.pop("answers", None)
        row.pop("metadata", None)
        row.pop("conversations", None)
        row.pop("independent_qa", None)
    return list(grouped.values())


def main() -> None:
    existing = read_existing() if ALL.exists() else {}

    train_rows: list[dict] = []
    if ALL.exists():
        with ALL.open() as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("source_split") == "train":
                    train_rows.append(row)
    else:
        # Permit schema migration of an already materialized 1qa-train.jsonl
        # when the intermediate 1qa-all.jsonl is not retained.
        current_train = ROOT / "1qa-train.jsonl"
        if not current_train.exists():
            raise FileNotFoundError(ALL)
        with current_train.open() as fh:
            train_rows = [json.loads(line) for line in fh if line.strip()]

    validation_rows: list[dict] = []
    manifest: dict[str, dict] = {}
    for dataset in DATASETS:
        target = VALIDATION_SPLITS[dataset]
        rows = existing.get((dataset, target), [])
        if not rows and not ALL.exists() and target == "validation":
            current_validation = ROOT / "1qa-validation.jsonl"
            if current_validation.exists():
                with current_validation.open() as fh:
                    rows = []
                    for line in fh:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if row.get("dataset") == dataset:
                            rows.append(row)
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

    # Keep one record per context so the writer/cache is computed once while the
    # collator can sample several QA turns from that context.
    train_rows = aggregate_context_rows(train_rows)
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
