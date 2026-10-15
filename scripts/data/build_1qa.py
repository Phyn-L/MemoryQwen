#!/usr/bin/env python3
"""Build the 1QA corpus from the local Hugging Face cache.

This entry point writes only the **raw, unfiltered** corpus under ``--root`` (default
``contexts/1qa``, the tree the configs and the evaluation suites read):

* ``meta/meta_<split>.json`` -- the merged split, one line per context with **all** of its QA
  pairs (JSON Lines content; ``scripts/data/build_cache.py`` filters it);
* ``meta/<split>/<slug>.json`` -- the same rows per dataset (``src/eval_data.py`` owns those
  directory names, so builder and readers cannot disagree);
* ``meta/build.manifest.json`` -- context/QA counts per dataset and split, plus which datasets
  have no such split at all.

The filtered products a run actually trains on (``cache/train.jsonl``,
``cache/validation.jsonl``) are built from these files by ``scripts/data/build_cache.py``.

Conventions (see ``contexts/README.md``):

* each line is ONE context with every QA pair that belongs to it; a context never borrows text
  from another row, and questions that share a context are aggregated into that one line;
* **no filtering**: no context-length cap and no per-dataset row cap. Only structurally
  meaningless samples are dropped -- no context, no question text, or a QA pair without a
  single non-empty answer;
* MS MARCO v1.1/v2.1 contexts are built from the ``is_selected=1`` passages only.

Sources, all local (nothing is downloaded):

* ``<cache>/hub/datasets--<org>--<name>/snapshots/<rev>/<config>/<split>-*.parquet`` for
  MS MARCO, RACE, DuoRC, HotpotQA and 2WikiMultihopQA;
* ``<cache>/datasets/squad/plain_text/0.0.0/*/squad-<split>.arrow`` for SQuAD (the legacy
  ``datasets`` arrow cache, read as an Arrow IPC stream);
* the MuSiQue JSONL files in their snapshot directory.

Usage::

    python scripts/data/build_1qa.py                      # every dataset, train/validation/test
    python scripts/data/build_1qa.py --datasets squad race --splits validation --dry-run
    python scripts/data/build_cache.py --config configs/train_baseline.yaml   # cache/ files
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval_data import DATASETS as EVAL_DATASETS  # noqa: E402
from src.eval_data import SPLIT_DIRS, slug  # noqa: E402

DEFAULT_CACHE = Path("/data/lz/hf_cache")
DEFAULT_ROOT = Path("contexts/1qa")
#: One task per (dataset, split) is independent, so the conversion runs in a process pool; the
#: parent collects the results in dataset order, which keeps the merged file deterministic.
DEFAULT_BUILD_WORKERS = max(1, min(8, os.cpu_count() or 1))
#: Raw rows live here; the filtered ``cache/`` tree is ``scripts/data/build_cache.py``'s job.
META_DIRNAME = "meta"
SPLITS = ("train", "validation", "test")

#: Per-dataset source description. ``convert`` names the row converter below.
SPECS: dict[str, dict[str, Any]] = {
    "squad": {
        "source": "arrow",
        "files": {
            "train": "datasets/squad/plain_text/0.0.0/*/squad-train.arrow",
            "validation": "datasets/squad/plain_text/0.0.0/*/squad-validation.arrow",
        },
        "convert": "squad",
    },
    "msmarcov1.1": {
        "source": "parquet", "repo": "microsoft/ms_marco", "config": "v1.1", "convert": "msmarco",
    },
    "msmarcov2.1": {
        "source": "parquet", "repo": "microsoft/ms_marco", "config": "v2.1", "convert": "msmarco",
    },
    "race": {"source": "parquet", "repo": "ehovy/race", "config": "all", "convert": "race"},
    "duorc/selfrc": {
        "source": "parquet", "repo": "ibm-research/duorc", "config": "SelfRC", "convert": "duorc",
    },
    "duorc/paraphraserc": {
        "source": "parquet", "repo": "ibm-research/duorc", "config": "ParaphraseRC",
        "convert": "duorc",
    },
    "hotpotqa(distractor)": {
        "source": "parquet", "repo": "hotpotqa/hotpot_qa", "config": "distractor",
        "convert": "hotpot",
    },
    "2wikimultihopqa": {
        "source": "parquet", "repo": "framolfese/2WikiMultihopQA", "config": "data",
        "convert": "hotpot",
    },
    "musique": {
        "source": "jsonl",
        "repo": "dgslibisey/MuSiQue",
        "files": {
            "train": "musique_ans_v1.0_train.jsonl",
            "validation": "musique_ans_v1.0_dev.jsonl",
        },
        "convert": "musique",
    },
}
BUILD_DATASETS = tuple(EVAL_DATASETS)


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _references(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    seen, out = set(), []
    for value in values or ():
        text = _clean(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _pair(index: int, dataset: str, split: str, question: str, answers: Any,
          **metadata) -> dict | None:
    """One QA pair in the stored schema, or None when it carries no usable QA."""
    question, answers = _clean(question), _references(answers)
    if not question or not answers:
        return None
    return {
        "id": f"{dataset}:{split}:{index}",
        "question": question,
        "answers": answers,
        "answer": answers[0],
        "metadata": {key: value for key, value in metadata.items() if value not in (None, "", [])},
    }


# --------------------------------------------------------------------------- converters
def convert_squad(dataset, split, row, index):
    """SQuAD: ``answers.text`` carries every annotator's span."""
    answers = row.get("answers") or {}
    return row.get("context", ""), _pair(
        index, dataset, split, row.get("question", ""), answers.get("text", []),
        title=row.get("title"),
    )


def convert_msmarco(dataset, split, row, index):
    """MS MARCO: the context is the *selected* passages only."""
    passages = row.get("passages") or {}
    selected = [
        _clean(text) for text, flag in zip(
            passages.get("passage_text", []), passages.get("is_selected", []), strict=False,
        ) if flag and _clean(text)
    ]
    return "\n\n".join(selected), _pair(
        index, dataset, split, row.get("query", ""), row.get("answers", []),
        query_type=row.get("query_type"), selected_passages=len(selected),
    )


def convert_race(dataset, split, row, index):
    """RACE: multiple choice; the answer letter must point at one of the four options."""
    options = [_clean(option) for option in (row.get("options") or [])]
    letter = _clean(row.get("answer")).upper()
    position = {"A": 0, "B": 1, "C": 2, "D": 3}.get(letter, -1)
    if len(options) != 4 or not 0 <= position < len(options):
        return row.get("article", ""), None
    return row.get("article", ""), _pair(
        index, dataset, split, row.get("question", ""), [options[position]],
        options=options, answer_letter=letter,
    )


def convert_duorc(dataset, split, row, index):
    """DuoRC: the plot is the context; ``no_answer`` rows carry no reference."""
    if row.get("no_answer"):
        return row.get("plot", ""), None
    return row.get("plot", ""), _pair(
        index, dataset, split, row.get("question", ""), row.get("answers", []),
        title=row.get("title"), plot_id=row.get("plot_id"),
    )


def convert_hotpot(dataset, split, row, index):
    """HotpotQA / 2WikiMultihopQA: the row's own paragraphs are its context."""
    context = row.get("context") or {}
    joiner = "".join if dataset.startswith("hotpot") else " ".join
    blocks = []
    for title, sentences in zip(
        context.get("title", []), context.get("sentences", []), strict=False,
    ):
        text = joiner(_clean(sentence) for sentence in (sentences or [])).strip()
        if text:
            blocks.append(f"{_clean(title)}: {text}" if _clean(title) else text)
    extra = (
        {"type": row.get("type"), "level": row.get("level")}
        if dataset.startswith("hotpot")
        else {"type": row.get("type"), "supporting_facts": row.get("supporting_facts")}
    )
    return "\n\n".join(blocks), _pair(
        index, dataset, split, row.get("question", ""), [row.get("answer", "")], **extra,
    )


def convert_musique(dataset, split, row, index):
    """MuSiQue: the provided paragraphs are the context; aliases are extra references."""
    blocks = [_clean(paragraph.get("paragraph_text", "")) for paragraph in row.get("paragraphs", [])]
    answers = [row.get("answer", ""), *(row.get("answer_aliases") or [])]
    return "\n\n".join(block for block in blocks if block), _pair(
        index, dataset, split, row.get("question", ""), answers,
        answerable=row.get("answerable"),
    )


#: When a split yields no usable row, try the next one. 2WikiMultihopQA's test split carries
#: questions whose ``answer`` field is empty, so its test set is its validation set -- the
#: manifest records which split was actually used.
SPLIT_FALLBACKS: dict[str, tuple[str, ...]] = {
    "test": ("validation", "train"),
    "validation": ("train",),
    "train": (),
}

CONVERTERS = {
    "squad": convert_squad,
    "msmarco": convert_msmarco,
    "race": convert_race,
    "duorc": convert_duorc,
    "hotpot": convert_hotpot,
    "musique": convert_musique,
}


# ------------------------------------------------------------------------------ reading
def _snapshot(cache: Path, repo: str) -> Path:
    """The snapshot directory of one ``datasets--<org>--<name>`` cache entry."""
    name = "datasets--" + repo.replace("/", "--")
    directory = cache / "hub" / name
    if not directory.is_dir():
        raise FileNotFoundError(f"no Hugging Face cache entry for {repo}: {directory}")
    ref = directory / "refs" / "main"
    if ref.is_file():
        snapshot = directory / "snapshots" / ref.read_text().strip()
        if snapshot.is_dir():
            return snapshot
    snapshots = sorted(path for path in (directory / "snapshots").glob("*") if path.is_dir())
    if len(snapshots) != 1:
        raise FileNotFoundError(
            f"{repo}: expected one snapshot under {directory / 'snapshots'}, found {snapshots}"
        )
    return snapshots[0]


def read_split(dataset: str, split: str, cache: Path) -> Iterator[dict]:
    """Yield the raw rows of one dataset/split from the local cache."""
    spec = SPECS[dataset]
    if spec["source"] == "parquet":
        snapshot = _snapshot(cache, spec["repo"])
        directory = snapshot / spec["config"] if spec["config"] else snapshot
        paths = sorted(directory.glob(f"{split}-*.parquet"))
        if not paths:
            raise FileNotFoundError(
                f"{dataset}: no {split} parquet under {directory} (the cache has the other splits)"
            )
        import pyarrow.parquet as pq

        for path in paths:
            yield from pq.read_table(path).to_pylist()
        return
    if spec["source"] == "jsonl":
        filename = spec["files"].get(split)
        if not filename:
            raise FileNotFoundError(f"{dataset}: the cache has no {split} split")
        path = _snapshot(cache, spec["repo"]) / filename
        if not path.is_file():
            raise FileNotFoundError(f"{dataset} {split}: {path} is missing")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    if spec["source"] == "arrow":
        pattern = spec["files"].get(split)
        if not pattern:
            raise FileNotFoundError(f"{dataset}: the cache has no {split} split")
        paths = sorted(cache.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"{dataset} {split}: no file matching {pattern} under {cache}")
        import pyarrow as pa
        import pyarrow.ipc as ipc

        for path in paths:
            # The legacy `datasets` cache writes Arrow IPC *streams*, not Arrow files.
            with pa.memory_map(str(path), "r") as source:
                yield from ipc.open_stream(source).read_all().to_pylist()
        return
    raise ValueError(f"{dataset}: unknown source {spec['source']!r}")


# -------------------------------------------------------------------------- aggregation
def convert_rows(dataset: str, split: str, raw_rows: Iterable[dict]) -> list[dict]:
    """Aggregate the QA rows of one dataset/split into one record per context.

    Unfiltered by design: the only rows that disappear are those without a context, without
    question text or without a single non-empty answer (see the module docstring).
    """
    converter = CONVERTERS[SPECS[dataset]["convert"]]
    grouped: OrderedDict[str, dict] = OrderedDict()
    for index, raw in enumerate(raw_rows):
        context, pair = converter(dataset, split, raw, index)
        context = _clean(context) if "\n" not in str(context or "") else _clean_block(context)
        if not context or pair is None:
            continue
        row = grouped.get(context)
        if row is None:
            row = {
                "id": pair["id"],
                "dataset": dataset,
                "source_split": split,
                "context": context,
                "qa_pairs": [],
                "metadata": {},
            }
            grouped[context] = row
        row["qa_pairs"].append(pair)
    return list(grouped.values())


def _clean_block(value: Any) -> str:
    """Trim a multi-paragraph context without collapsing its internal line breaks."""
    lines = [line.strip() for line in str(value or "").splitlines()]
    return "\n".join(lines).strip()


# ------------------------------------------------------------------------------ writing
def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_corpus(root: Path, rows_by_split: dict[str, dict[str, list[dict]]]) -> dict:
    """Write the merged raw splits, the per-dataset raw files and the manifest."""
    root = Path(root) / META_DIRNAME
    manifest: dict[str, Any] = {"built_by": "scripts/data/build_1qa.py", "splits": {}}
    missing: dict[str, list[str]] = {}
    for split in SPLITS:
        per_dataset = rows_by_split.get(split, {})
        merged_rows = [
            row for dataset in BUILD_DATASETS for row in per_dataset.get(dataset, ())
        ]
        if not merged_rows:
            missing[split] = [dataset for dataset in BUILD_DATASETS]
            continue
        merged_path = root / f"meta_{split}.json"
        _write_jsonl(merged_path, merged_rows)
        details = {}
        for dataset in BUILD_DATASETS:
            rows = per_dataset.get(dataset)
            if not rows:
                continue
            details[dataset] = {
                "contexts": len(rows),
                "qa_pairs": sum(len(row["qa_pairs"]) for row in rows),
                "requested_split": split,
                "actual_source_split": rows[0]["source_split"],
                "file": None,
            }
            if split in SPLIT_DIRS:
                path = root / SPLIT_DIRS[split] / f"{slug(dataset)}.json"
                _write_jsonl(path, rows)
                details[dataset]["file"] = str(path.relative_to(root))
        manifest["splits"][split] = {
            "file": merged_path.name,
            "contexts": len(merged_rows),
            "qa_pairs": sum(len(row["qa_pairs"]) for row in merged_rows),
            "datasets": details,
        }
        absent = [dataset for dataset in BUILD_DATASETS if dataset not in details]
        if absent:
            missing[split] = absent
    if missing:
        manifest["missing_splits"] = missing
    (root / "build.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def read_dataset_split(dataset: str, split: str, cache: Path) -> tuple[list[dict], str]:
    """``(rows, actual_split)`` for one dataset, following :data:`SPLIT_FALLBACKS`.

    Returns an empty list and the requested split when neither the split nor any fallback has
    a usable row, so the caller can report it as missing instead of writing an empty file.
    """
    for candidate in (split, *SPLIT_FALLBACKS.get(split, ())):
        try:
            rows = convert_rows(dataset, candidate, read_split(dataset, candidate, cache))
        except FileNotFoundError:
            continue
        if rows:
            return rows, candidate
        # The split exists but holds nothing usable (empty answers, no selected passage):
        # try the next fallback rather than writing an empty file.
    return [], split


@dataclass
class _Task:
    """One ``(dataset, split)`` to read and convert; picklable, so it can be a pool task."""

    dataset: str
    split: str
    cache: str


def _run_task(task: _Task) -> tuple[str, str, list[dict], str]:
    """Convert one dataset/split into ``(dataset, split, rows, note)``."""
    try:
        rows, actual = read_dataset_split(task.dataset, task.split, Path(task.cache))
    except Exception as error:  # noqa: BLE001 - reported by the parent, never swallowed
        return task.dataset, task.split, [], f"{type(error).__name__}: {error}"
    if actual != task.split:
        return task.dataset, task.split, rows, f" (from {actual})"
    if not rows:
        return task.dataset, task.split, [], f"no usable rows in {task.split} or its fallbacks"
    return task.dataset, task.split, rows, ""


def _convert_tasks(tasks: Iterable[_Task], num_workers: int):
    """Yield task results in submission order (which is the parent's write order)."""
    ordered = list(tasks)
    if num_workers > 1 and len(ordered) > 1:
        with ProcessPoolExecutor(max_workers=min(int(num_workers), len(ordered))) as executor:
            yield from executor.map(_run_task, ordered)
        return
    for task in ordered:
        yield _run_task(task)


def build(cache: Path, root: Path, datasets: Iterable[str], splits: Iterable[str],
          dry_run: bool = False, quiet: bool = False,
          num_workers: int = DEFAULT_BUILD_WORKERS) -> dict:
    """Read every requested dataset/split, convert it and (unless dry-run) write the corpus.

    The conversion is CPU-bound (parquet decoding plus JSON/dict work) and one
    ``(dataset, split)`` is independent of the others, so the tasks go through a process pool;
    the parent collects them in dataset order, so a parallel build writes the same bytes as a
    serial one.
    """
    wanted_splits = [split for split in SPLITS if split in set(splits)]
    wanted_datasets = list(datasets)
    tasks = [
        _Task(dataset, split, str(cache))
        for split in wanted_splits
        for dataset in wanted_datasets
    ]
    rows_by_split: dict[str, dict[str, list[dict]]] = {split: {} for split in wanted_splits}
    for dataset, split, rows, note in _convert_tasks(tasks, int(num_workers)):
        if not rows:
            if not quiet:
                print(f"[skip] {dataset} {split}: {note.strip()}", flush=True)
            continue
        rows_by_split[split][dataset] = rows
        if not quiet:
            pairs = sum(len(row["qa_pairs"]) for row in rows)
            print(
                f"[read] {dataset:22s} {split:10s} contexts={len(rows):7d} "
                f"qa_pairs={pairs:8d}{note}",
                flush=True,
            )
    if dry_run:
        return {
            split: {
                dataset: {
                    "contexts": len(rows),
                    "qa_pairs": sum(len(row["qa_pairs"]) for row in rows),
                }
                for dataset, rows in per_dataset.items()
            }
            for split, per_dataset in rows_by_split.items()
        }
    manifest = write_corpus(Path(root), rows_by_split)
    if not quiet:
        print(json.dumps(manifest["splits"], ensure_ascii=False, indent=2), flush=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                        help="Hugging Face cache root (hub/ and legacy datasets/ trees)")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Output root; raw splits are written under <root>/meta/")
    parser.add_argument("--datasets", nargs="+", choices=BUILD_DATASETS,
                        default=list(BUILD_DATASETS))
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the converted counts without writing anything")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_BUILD_WORKERS,
                        help="Worker processes for the per-dataset conversion (1 = serial)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    result = build(
        args.cache, args.root, args.datasets, args.splits, args.dry_run, args.quiet,
        num_workers=args.num_workers,
    )
    if args.dry_run and not args.quiet:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
