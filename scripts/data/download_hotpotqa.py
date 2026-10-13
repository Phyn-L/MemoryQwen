#!/usr/bin/env python3
"""Download HotpotQA from Hugging Face and write the aggregated JSONL schema.

The default ``distractor`` configuration contains the ten candidate Wikipedia
paragraphs supplied with each question.  Questions sharing the same rendered
context are grouped into one context record, matching the layout under
``/data/lz/contexts/aggregated``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET = "hotpotqa/hotpot_qa"
DEFAULT_CONFIG = "distractor"
DEFAULT_OUTPUT = Path("/data/lz/contexts/aggregated")
DEFAULT_CACHE = Path(os.environ.get("HF_HOME", "/data/lz/hf_cache"))
OUTPUT_SPLITS = ("train", "validation", "test")


def render_context(context: dict[str, Any]) -> str:
    """Render HotpotQA's title/sentence lists as one readable context string."""
    titles = context.get("title") or []
    sentence_groups = context.get("sentences") or []
    if len(titles) != len(sentence_groups):
        raise ValueError(
            "HotpotQA context has mismatched title/sentence lengths: "
            f"{len(titles)} != {len(sentence_groups)}"
        )

    sections = []
    for title, sentences in zip(titles, sentence_groups):
        body = " ".join(str(sentence).strip() for sentence in sentences or ()).strip()
        if body:
            sections.append(f"{str(title).strip()}\n{body}")
    return "\n\n".join(sections).strip()


def make_qa(row: dict[str, Any], split: str, config: str) -> dict[str, Any]:
    question = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()
    qa_id = str(row.get("id", "")).strip()
    if not qa_id or not question or not answer:
        raise ValueError(f"HotpotQA row has an empty id, question, or answer: {row!r}")

    supporting_facts = row.get("supporting_facts") or {}
    context = row.get("context") or {}
    return {
        "id": qa_id,
        "question": question,
        "answers": [answer],
        "answer_starts": [],
        "metadata": {
            "type": row.get("type"),
            "level": row.get("level"),
            "supporting_facts": {
                "title": list(supporting_facts.get("title") or []),
                "sent_id": list(supporting_facts.get("sent_id") or []),
            },
            "context_titles": list(context.get("title") or []),
            "config": config,
        },
        "dataset": "hotpotqa",
        "split": split,
        "source_dataset": "hotpotqa",
        "source_split": split,
    }


def aggregate_split(rows: Iterable[dict[str, Any]], split: str, config: str) -> OrderedDict[str, dict[str, Any]]:
    """Group source rows by exact rendered context while preserving QA order."""
    contexts: OrderedDict[str, dict[str, Any]] = OrderedDict()
    seen_qa_ids: set[str] = set()
    for row in rows:
        context = render_context(row.get("context") or {})
        if not context:
            raise ValueError(f"HotpotQA row {row.get('id')!r} has an empty context")
        context_id = hashlib.sha256(context.encode("utf-8")).hexdigest()
        qa = make_qa(row, split, config)
        if qa["id"] in seen_qa_ids:
            raise ValueError(f"Duplicate HotpotQA question id: {qa['id']}")
        seen_qa_ids.add(qa["id"])

        record = contexts.setdefault(
            context_id,
            {
                "context_id": context_id,
                "context": context,
                "qa_pairs": [],
            },
        )
        record["qa_pairs"].append(qa)
    return contexts


def finalize_record(record: dict[str, Any], qa_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    source_splits = sorted({qa["source_split"] for qa in qa_pairs})
    return {
        "context_id": record["context_id"],
        "context": record["context"],
        "qa_pairs": qa_pairs,
        "num_qa_pairs": len(qa_pairs),
        "source_datasets": ["hotpotqa"],
        "source_splits": source_splits,
        "source_versions": [],
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def convert(
    dataset_name: str,
    config: str,
    output_dir: Path,
    cache_dir: Path,
    overwrite: bool,
) -> dict[str, dict[str, int]]:
    from datasets import load_dataset

    target_dir = output_dir / "hotpotqa"
    if target_dir.exists() and any(target_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {target_dir}. Use --overwrite to replace its files."
        )

    dataset = load_dataset(dataset_name, config, cache_dir=str(cache_dir))
    if "train" not in dataset or "validation" not in dataset:
        raise ValueError(f"Expected train and validation splits, got {list(dataset)}")

    all_contexts: dict[str, dict[str, Any]] = {}
    split_contexts: dict[str, OrderedDict[str, dict[str, Any]]] = {}
    for split in ("train", "validation"):
        current = aggregate_split(dataset[split], split, config)
        split_contexts[split] = current
        for context_id, record in current.items():
            if context_id not in all_contexts:
                all_contexts[context_id] = {
                    "context_id": context_id,
                    "context": record["context"],
                    "qa_pairs": [],
                }
            all_contexts[context_id]["qa_pairs"].extend(record["qa_pairs"])

    canonical = [
        finalize_record(record, record["qa_pairs"])
        for _, record in sorted(all_contexts.items())
    ]
    split_records = {
        split: [
            finalize_record(record, record["qa_pairs"])
            for _, record in sorted(split_contexts[split].items())
        ]
        for split in ("train", "validation")
    }
    split_records["test"] = []

    target_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "contexts": {"contexts": write_jsonl(target_dir / "contexts.jsonl", canonical)},
        "splits": {},
    }
    for split in OUTPUT_SPLITS:
        records = split_records[split]
        counts["splits"][split] = {
            "contexts": write_jsonl(target_dir / f"{split}.jsonl", records),
            "qa_pairs": sum(record["num_qa_pairs"] for record in records),
        }
    counts["contexts"]["qa_pairs"] = sum(record["num_qa_pairs"] for record in canonical)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset name")
    parser.add_argument("--config", choices=("distractor", "fullwiki"), default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing HotpotQA output files")
    args = parser.parse_args()

    counts = convert(args.dataset, args.config, args.output_dir, args.cache_dir, args.overwrite)
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
