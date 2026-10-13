#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from split_c1qa_eval_sets import _snapshot, cache_rows, make_row


ROOT = Path("data/generated")
OUT = ROOT / "1qa-test.jsonl"
MSMARCO_ROOT = Path("/data/lz/hf_cache/hub")


def msmarco_v11_test() -> list[dict]:
    snap = _snapshot(MSMARCO_ROOT, "microsoft/ms_marco")
    paths = sorted((snap / "v1.1").glob("test-*.parquet"))
    rows = []
    for path in paths:
        for i, raw in enumerate(pq.read_table(path).to_pylist()):
            row = make_row("msmarcov1.1", "test", raw, i)
            if row:
                rows.append(row)
    return rows


def read_file(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def main() -> None:
    # Existing files already contain the requested 1024 examples for all other
    # datasets. Replace only MS MARCO v1.1 with answerable test examples.
    v11 = msmarco_v11_test()
    if len(v11) < 1024:
        raise RuntimeError(f"MS MARCO v1.1 test has only {len(v11)} answerable rows")
    v11 = v11[:1024]
    v11_path = ROOT / "1qa-test/msmarcov1.1.jsonl"
    with v11_path.open("w") as fh:
        for row in v11:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    names = [
        ("squad", ROOT / "1qa-test/squad.jsonl", "validation"),
        ("msmarcov1.1", v11_path, "test"),
        ("msmarcov2.1", ROOT / "1qa-test/msmarcov2.1.jsonl", "validation"),
        ("race", ROOT / "1qa-test/race.jsonl", "test"),
        ("hotpotqa(distractor)", ROOT / "1qa-test/hotpotqa_distractor.jsonl", "validation"),
        ("musique", ROOT / "1qa-test/musique.jsonl", "dev"),
        ("2wikimultihopqa", ROOT / "1qa-test/2wikimultihopqa.jsonl", "validation"),
    ]
    manifest = {}
    with OUT.open("w") as out:
        for dataset, path, source_split in names:
            rows = read_file(path)
            if len(rows) != 1024:
                raise RuntimeError(f"{path} has {len(rows)} rows")
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest[dataset] = {
                "rows": len(rows),
                "source_split": source_split,
                "file": str(path),
            }
    manifest["total_rows"] = 7 * 1024
    (ROOT / "1qa-test.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
