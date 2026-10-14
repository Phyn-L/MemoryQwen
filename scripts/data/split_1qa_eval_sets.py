#!/usr/bin/env python3
"""Create fixed-size per-dataset C1QA test and validation subsets."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from build_1qa_from_hf_cache import _clean, _record, _snapshot


CACHE = Path("/data/lz/hf_cache")
ROOT = Path("data/generated")
ALL = ROOT / "1qa-all.jsonl"
TEST_DATASETS = {
    "squad": "validation",
    "msmarcov1.1": "validation",
    "msmarcov2.1": "test",
    "race": "test",
    "hotpotqa(distractor)": "validation",
    "musique": "dev",
    "2wikimultihopqa": "test",
}
VAL_DATASETS = {
    "squad": "validation",
    "msmarcov1.1": "validation",
    "msmarcov2.1": "validation",
    "race": "validation",
    "duorc/selfrc": "validation",
    "duorc/paraphraserc": "validation",
    "hotpotqa(distractor)": "validation",
    "musique": "dev",
    "2wikimultihopqa": "validation",
}


def rows(path: Path):
    yield from pq.read_table(path).to_pylist()


def make_row(dataset: str, split: str, row: dict[str, Any], index: int):
    if dataset.startswith("msmarco"):
        p = row.get("passages") or {}
        if not any(p.get("is_selected", [])) or not row.get("answers"):
            return None
        return _record(dataset, split, row.get("query_id", str(index)),
                       "\n\n".join(_clean(x) for x in p.get("passage_text", []) if _clean(x)),
                       row.get("query", ""), row.get("answers", []),
                       {"query_type": row.get("query_type"), "selected_passages": p.get("is_selected", [])})
    if dataset == "race":
        opts, letter = row.get("options") or [], _clean(row.get("answer")).upper()
        idx = {"A": 0, "B": 1, "C": 2, "D": 3}.get(letter, -1)
        if idx < 0 or idx >= len(opts):
            return None
        return _record(dataset, split, row.get("example_id", str(index)), row.get("article", ""),
                       row.get("question", ""), [opts[idx]], {"answer_letter": letter, "options": opts})
    if dataset.startswith("duorc"):
        if row.get("no_answer") or not row.get("answers"):
            return None
        return _record(dataset, split, row.get("question_id", str(index)), row.get("plot", ""),
                       row.get("question", ""), row.get("answers", []),
                       {"plot_id": row.get("plot_id"), "title": row.get("title", "")})
    if dataset == "musique":
        blocks = [p.get("paragraph_text", "") for p in row.get("paragraphs", [])]
        return _record(
            dataset,
            split,
            row.get("id", str(index)),
            "\n\n".join(_clean(x) for x in blocks if _clean(x)),
            row.get("question", ""),
            [row.get("answer", "")],
            {"answerable": row.get("answerable"), "answer_aliases": row.get("answer_aliases", [])},
        )
    if dataset in {"hotpotqa(distractor)", "2wikimultihopqa"}:
        c = row.get("context") or {}
        blocks = []
        for title, sentences in zip(c.get("title", []), c.get("sentences", [])):
            text = ("".join if dataset.startswith("hotpot") else " ".join)(
                _clean(x) for x in (sentences or [])
            ).strip()
            if text:
                blocks.append(f"{title}: {text}" if title else text)
        return _record(dataset, split, row.get("id", str(index)), "\n\n".join(blocks),
                       row.get("question", ""), [row.get("answer", "")],
                       {"type": row.get("type"), "level": row.get("level")} if dataset.startswith("hotpot")
                       else {"type": row.get("type"), "supporting_facts": row.get("supporting_facts")})
    return None


def cache_rows(dataset: str, split: str):
    hub = CACHE / "hub"
    if dataset == "squad":
        # SQuAD validation is already in 1qa-all; handled by existing rows.
        return []
    if dataset.startswith("msmarco"):
        snap = _snapshot(hub, "microsoft/ms_marco")
        version = "v1.1" if dataset == "msmarcov1.1" else "v2.1"
        paths = sorted((snap / version).glob(f"{split}-*.parquet")) if snap else []
    elif dataset == "race":
        snap = _snapshot(hub, "ehovy/race")
        paths = sorted((snap / "all").glob(f"{split}-*.parquet")) if snap else []
    elif dataset.startswith("duorc"):
        snap = _snapshot(hub, "ibm-research/duorc")
        config = "SelfRC" if dataset.endswith("selfrc") else "ParaphraseRC"
        paths = sorted((snap / config).glob(f"{split}-*.parquet")) if snap else []
    elif dataset == "hotpotqa(distractor)":
        snap = _snapshot(hub, "hotpotqa/hotpot_qa")
        paths = sorted((snap / "distractor").glob(f"{split}-*.parquet")) if snap else []
    elif dataset == "2wikimultihopqa":
        snap = _snapshot(hub, "framolfese/2WikiMultihopQA")
        paths = sorted((snap / "data").glob(f"{split}-*.parquet")) if snap else []
    elif dataset == "musique":
        snap = _snapshot(hub, "dgslibisey/MuSiQue")
        filename = "musique_ans_v1.0_dev.jsonl" if split == "dev" else "musique_ans_v1.0_train.jsonl"
        p = snap / filename if snap else None
        if not p or not p.exists():
            return []
        return [json.loads(line) for line in p.read_text().splitlines()]
    else:
        return []
    return [row for path in paths for row in rows(path)]


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if not ALL.exists():
        old = Path("/data/lz/contexts/ift_c1qa.jsonl")
        if not old.exists():
            raise FileNotFoundError("Neither generated C1QA nor old source file exists")
        os.rename(old, ALL)
    existing = {}
    with ALL.open() as fh:
        for line in fh:
            row = json.loads(line)
            existing.setdefault((row["dataset"], row["source_split"]), []).append(row)

    manifests = {}
    for folder, plan in (("1qa-test", TEST_DATASETS), ("1qa-val", VAL_DATASETS)):
        outdir = ROOT / folder
        outdir.mkdir(exist_ok=True)
        for dataset, preferred in plan.items():
            selected = existing.get((dataset, preferred), [])
            actual = preferred
            if not selected:
                raw = cache_rows(dataset, preferred)
                selected = [make_row(dataset, preferred, r, i) for i, r in enumerate(raw)]
                selected = [r for r in selected if r]
                if selected:
                    actual = preferred
            if not selected:
                fallback = "validation" if preferred not in {"validation", "dev"} else "train"
                selected = existing.get((dataset, fallback), [])
                actual = fallback
                if not selected:
                    raw = cache_rows(dataset, fallback)
                    selected = [make_row(dataset, fallback, r, i) for i, r in enumerate(raw)]
                    selected = [r for r in selected if r]
            if len(selected) < 1024:
                raise RuntimeError(f"{dataset} {folder} has only {len(selected)} rows from {actual}")
            selected = selected[:1024]
            path = outdir / f"{dataset.replace('/', '_').replace('(', '_').replace(')', '')}.jsonl"
            with path.open("w") as fh:
                for row in selected:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifests.setdefault(folder, {})[dataset] = {
                "requested": preferred, "actual_source_split": actual,
                "rows": len(selected), "path": str(path),
            }
    (ROOT / "1qa-splits.manifest.json").write_text(json.dumps(manifests, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifests, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
