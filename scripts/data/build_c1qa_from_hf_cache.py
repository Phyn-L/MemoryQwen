#!/usr/bin/env python3
"""Build SHINE-style C1QA records from the local Hugging Face cache only.

The script intentionally never falls back to /data/lz/contexts/raw and never
contacts the Hub.  Missing source shards are reported in the manifest.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


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


def _clean(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _snapshot(root: Path, repo: str) -> Path | None:
    candidates = [
        root / ("datasets--" + repo.replace("/", "--")) / "snapshots",
        # Some cached dataset IDs omit the organization for legacy builders.
        root / ("datasets--" + repo.split("/")[-1]) / "snapshots",
    ]
    for base in candidates:
        snapshots = sorted(p for p in base.glob("*") if p.is_dir())
        if snapshots:
            return snapshots[-1]
    return None


def _parquet_rows(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        table = pq.read_table(path)
        yield from table.to_pylist()


def _record(
    dataset: str,
    source_split: str,
    row_id: str,
    context: str,
    question: str,
    answers: list[str],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    context, question = _clean(context), _clean(question)
    answers = [_clean(a) for a in answers if _clean(a)]
    if not context or not question or not answers:
        return None
    answer = answers[0]
    return {
        "id": f"{dataset}:{source_split}:{row_id}",
        "dataset": dataset,
        "source_split": source_split,
        "context": context,
        "question": question,
        "answers": answers,
        "metadata": metadata or {},
        "conversations": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer[:1].upper() + answer[1:]},
        ],
    }


def _squad(cache: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base = cache / "datasets" / "squad"
    files = sorted(base.glob("**/squad-train.arrow"))
    files += sorted(base.glob("**/squad-validation.arrow"))
    if not files:
        return [], {"status": "missing", "reason": "squad Arrow files not found"}
    import pyarrow.ipc as ipc

    records: list[dict[str, Any]] = []
    for path in files:
        split = "train" if "train" in path.name else "validation"
        with path.open("rb") as fh:
            table = ipc.open_stream(fh).read_all()
        for row in table.to_pylist():
            answers = row.get("answers") or {}
            rec = _record(
                "squad",
                split,
                row.get("id", str(len(records))),
                row.get("context", ""),
                row.get("question", ""),
                answers.get("text", []) if isinstance(answers, dict) else [],
                {"title": row.get("title", "")},
            )
            if rec:
                records.append(rec)
    return records, {"status": "ok", "rows": len(records)}


def _duorc(snapshot: Path, config: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base = snapshot / config
    files = sorted(base.glob("train-*.parquet")) + sorted(base.glob("validation-*.parquet"))
    if not files:
        return [], {"status": "missing", "reason": f"{config} parquet files not found"}
    records = []
    for path in files:
        split = "train" if "/train-" in str(path) else "validation"
        for row in _parquet_rows([path]):
            if row.get("no_answer") or not row.get("answers"):
                continue
            rec = _record(
                f"duorc/{config.lower()}",
                split,
                row.get("question_id", row.get("plot_id", str(len(records)))),
                row.get("plot", ""),
                row.get("question", ""),
                row.get("answers", []),
                {"plot_id": row.get("plot_id"), "title": row.get("title", "")},
            )
            if rec:
                records.append(rec)
    return records, {"status": "ok", "rows": len(records)}


def _multihop(snapshot: Path, dataset: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files = sorted((snapshot / "data").glob("train-*.parquet"))
    if not files:
        return [], {"status": "missing", "reason": "train parquet files not found"}
    records = []
    for path in files:
        for row in _parquet_rows([path]):
            ctx = row.get("context") or {}
            titles = ctx.get("title", [])
            sentences = ctx.get("sentences", [])
            blocks = []
            for title, sents in zip(titles, sentences):
                text = " ".join(_clean(s) for s in (sents or [])).strip()
                if text:
                    blocks.append(f"{title}: {text}" if title else text)
            rec = _record(
                dataset,
                "train",
                row.get("id", str(len(records))),
                "\n\n".join(blocks),
                row.get("question", ""),
                [row.get("answer", "")],
                {"type": row.get("type"), "supporting_facts": row.get("supporting_facts")},
            )
            if rec:
                records.append(rec)
    return records, {"status": "ok", "rows": len(records)}


def _hotpot(snapshot: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files = sorted((snapshot / "distractor").glob("train-*.parquet"))
    if not files:
        return [], {"status": "missing", "reason": "distractor train parquet files not found"}
    records = []
    for row in _parquet_rows(files):
        ctx = row.get("context") or {}
        blocks = []
        for title, sents in zip(ctx.get("title", []), ctx.get("sentences", [])):
            text = "".join(_clean(s) for s in (sents or [])).strip()
            if text:
                blocks.append(f"{title}: {text}" if title else text)
        rec = _record(
            "hotpotqa(distractor)",
            "train",
            row.get("id", str(len(records))),
            "\n\n".join(blocks),
            row.get("question", ""),
            [row.get("answer", "")],
            {"type": row.get("type"), "level": row.get("level")},
        )
        if rec:
            records.append(rec)
    return records, {"status": "ok", "rows": len(records)}


def _musique(snapshot: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = sorted(snapshot.glob("musique_ans_v1.0_train.jsonl"))
    if not paths:
        return [], {"status": "missing", "reason": "MuSiQue train JSONL not found"}
    records = []
    for line in paths[0].read_text().splitlines():
        row = json.loads(line)
        blocks = [p.get("paragraph_text", "") for p in row.get("paragraphs", [])]
        rec = _record(
            "musique",
            "train",
            row.get("id", str(len(records))),
            "\n\n".join(_clean(x) for x in blocks if _clean(x)),
            row.get("question", ""),
            [row.get("answer", "")],
            {"answerable": row.get("answerable"), "answer_aliases": row.get("answer_aliases", [])},
        )
        if rec:
            records.append(rec)
    return records, {"status": "ok", "rows": len(records)}


def _msmarco(snapshot: Path, version: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base = snapshot / version
    files = sorted(base.glob("train-*.parquet")) + sorted(base.glob("validation-*.parquet"))
    if not files:
        return [], {"status": "missing", "reason": f"{version} parquet files not found"}
    records = []
    for path in files:
        split = "train" if "/train-" in str(path) else "validation"
        for row in _parquet_rows([path]):
            passages = row.get("passages") or {}
            selected = passages.get("is_selected", [])
            if not any(selected) or not row.get("answers"):
                continue
            context = "\n\n".join(_clean(x) for x in passages.get("passage_text", []) if _clean(x))
            rec = _record(
                f"msmarco{version}",
                split,
                row.get("query_id", str(len(records))),
                context,
                row.get("query", ""),
                row.get("answers", []),
                {"query_type": row.get("query_type"), "selected_passages": selected},
            )
            if rec:
                records.append(rec)
    return records, {"status": "ok", "rows": len(records)}


def _race(snapshot: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base = snapshot / "all"
    files = sorted(base.glob("train-*.parquet")) + sorted(base.glob("validation-*.parquet"))
    if not files:
        return [], {"status": "missing", "reason": "RACE all train/validation parquet files not found"}
    records = []
    answer_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    for path in files:
        split = "train" if "/train-" in str(path) else "validation"
        for row in _parquet_rows([path]):
            letter = _clean(row.get("answer")).upper()
            options = row.get("options") or []
            if letter not in answer_idx or answer_idx[letter] >= len(options):
                continue
            rec = _record(
                "race",
                split,
                row.get("example_id", str(len(records))),
                row.get("article", ""),
                row.get("question", ""),
                [options[answer_idx[letter]]],
                {"answer_letter": letter, "options": options},
            )
            if rec:
                records.append(rec)
    return records, {"status": "ok", "rows": len(records)}


def _unsupported(snapshot: Path | None, dataset: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return [], {
        "status": "missing",
        "reason": "Hub snapshot contains metadata only; no source data shards",
        "snapshot": str(snapshot) if snapshot else None,
        "dataset": dataset,
    }


def build(cache: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {"cache_root": str(cache), "datasets": {}}

    handlers = {
        "msmarcov1.1": lambda: _msmarco(_snapshot(cache / "hub", "microsoft/ms_marco"), "v1.1"),
        "msmarcov2.1": lambda: _msmarco(_snapshot(cache / "hub", "microsoft/ms_marco"), "v2.1"),
        "squad": lambda: _squad(cache),
        "race": lambda: _race(_snapshot(cache / "hub", "ehovy/race")),
        "duorc/selfrc": lambda: _duorc(_snapshot(cache / "hub", "ibm-research/duorc"), "SelfRC"),
        "duorc/paraphraserc": lambda: _duorc(_snapshot(cache / "hub", "ibm-research/duorc"), "ParaphraseRC"),
        "hotpotqa(distractor)": lambda: _hotpot(_snapshot(cache / "hub", "hotpotqa/hotpot_qa")),
        "musique": lambda: _musique(_snapshot(cache / "hub", "dgslibisey/MuSiQue")),
        "2wikimultihopqa": lambda: _multihop(_snapshot(cache / "hub", "framolfese/2WikiMultihopQA"), "2wikimultihopqa"),
    }
    for name in DATASETS:
        if name in handlers:
            records, info = handlers[name]()
        else:
            repo = "microsoft/ms_marco" if name.startswith("msmarco") else "ehovy/race"
            records, info = _unsupported(_snapshot(cache / "hub", repo), name)
        output.extend(records)
        manifest["datasets"][name] = info
    manifest["total_rows"] = len(output)
    manifest["missing"] = [k for k, v in manifest["datasets"].items() if v["status"] == "missing"]
    return output, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("/data/lz/hf_cache"))
    parser.add_argument("--output", type=Path, default=Path("data/generated/ift_c1qa_from_hf_cache.jsonl"))
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    records, manifest = build(args.cache_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
