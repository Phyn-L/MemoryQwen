from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

CACHE_VERSION = 1
CACHE_DIR_NAME = "dataset_cache"


def compute_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=list)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def source_files_metadata(files: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    """Identity of the source JSONL files, used to invalidate caches."""
    metadata = []
    for name, path in files:
        stat = path.stat()
        metadata.append(
            {
                "dataset": name,
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return metadata


def build_metadata(
    files: list[tuple[str, Path]],
    *,
    split: str,
    max_context_tokens: int,
    filter_long_context: bool,
    filter_no_qa: bool,
) -> dict[str, Any]:
    """Content identity of a filtered dataset.

    Deliberately excludes ``max_samples``: bounding a dataset does not change the
    filtered content, so capped and uncapped runs share one cache entry.
    """
    payload = {
        "version": CACHE_VERSION,
        "split": split,
        "files": source_files_metadata(files),
        "max_context_tokens": max_context_tokens,
        "filter_long_context": filter_long_context,
        "filter_no_qa": filter_no_qa,
    }
    payload["fingerprint"] = compute_fingerprint(payload)
    return payload


@contextmanager
def _lock(path: Path) -> Iterator[None]:
    with path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def load_records(
    cache_dir: Path,
    metadata: dict[str, Any],
) -> list[tuple[str, str, str, str, str]] | None:
    """Return cached ``(context, question, answer, dataset, context_id)`` tuples."""
    data_path = cache_dir / "records.jsonl"
    meta_path = cache_dir / "records.meta.json"
    if not data_path.exists() or not meta_path.exists():
        return None
    try:
        cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if cached_meta.get("fingerprint") != metadata["fingerprint"]:
            return None
        records: list[tuple[str, str, str, str, str]] = []
        with data_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                records.append(
                    (
                        str(row["context"]),
                        str(row["question"]),
                        str(row["answer"]),
                        str(row.get("dataset", "")),
                        str(row.get("context_id", "")),
                    )
                )
        if cached_meta.get("count") != len(records):
            return None
        return records
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None


def save_records(
    cache_dir: Path,
    metadata: dict[str, Any],
    records: list[tuple[str, str, str, str, str]],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = cache_dir / "records.jsonl"
    meta_path = cache_dir / "records.meta.json"
    data_tmp = data_path.with_name(f"{data_path.name}.{os.getpid()}.tmp")
    meta_tmp = meta_path.with_name(f"{meta_path.name}.{os.getpid()}.tmp")
    with data_tmp.open("w", encoding="utf-8") as handle:
        for context, question, answer, dataset, context_id in records:
            handle.write(
                json.dumps(
                    {
                        "context": context,
                        "question": question,
                        "answer": answer,
                        "dataset": dataset,
                        "context_id": context_id,
                    },
                    ensure_ascii=False,
                )
            )
            handle.write("\n")
    meta = dict(metadata)
    meta["count"] = len(records)
    meta_tmp.write_text(json.dumps(meta), encoding="utf-8")
    os.replace(data_tmp, data_path)
    os.replace(meta_tmp, meta_path)


def cached_load(
    cache_dir: Path,
    metadata: dict[str, Any],
    build,
    *,
    verbose: bool = True,
) -> list[tuple[str, str, str, str, str]]:
    """Read the cache or rebuild it under an exclusive lock."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    with _lock(cache_dir / "records.lock"):
        cached = load_records(cache_dir, metadata)
        if cached is not None:
            if verbose:
                print(f"Reusing dataset cache: {cache_dir / 'records.jsonl'} ({len(cached)} records)")
            return cached
        records = build()
        save_records(cache_dir, metadata, records)
        if verbose:
            print(f"Saved dataset cache: {cache_dir / 'records.jsonl'} ({len(records)} records)")
        return records
