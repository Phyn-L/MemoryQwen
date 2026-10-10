from __future__ import annotations
import fcntl, hashlib, json, os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from datasets import Dataset

CACHE_VERSION = 6  # Conversation schema replaces the legacy qa_pairs cache column.
CACHE_DIR_NAME = "dataset_cache"
DATASET_DIR_NAME = "hf_dataset"

def compute_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=list).encode()).hexdigest()

def source_files_metadata(files):
    return [{"dataset": n, "path": str(p.resolve()), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for n, p in files]

def build_metadata(files, *, split, max_context_tokens, filter_long_context, filter_no_qa, tokenizer=None):
    payload = {"version": CACHE_VERSION, "split": split, "files": source_files_metadata(files), "max_context_tokens": max_context_tokens, "filter_long_context": filter_long_context, "filter_no_qa": filter_no_qa}
    if tokenizer is not None:
        payload["tokenizer"] = {
            "name": str(getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)),
            "class": tokenizer.__class__.__name__,
            "vocab_size": getattr(tokenizer, "vocab_size", None),
        }
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is not None:
            payload["tokenizer"]["backend"] = hashlib.sha256(backend.to_str().encode()).hexdigest()
    payload["fingerprint"] = compute_fingerprint(payload)
    return payload

@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Exclusive advisory lock on ``path`` (the file is created if missing).

    Shared by both cache writers (the dataset cache here and the sortish-length cache in
    ``src/data.py``) so they use one locking protocol instead of two copies of the same
    three ``fcntl`` calls.
    """
    with path.open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

def load_records(cache_dir: Path, metadata: dict[str, Any]) -> Dataset | None:
    dataset_dir, meta_path = cache_dir / DATASET_DIR_NAME, cache_dir / "records.meta.json"
    if not dataset_dir.exists() or not meta_path.exists(): return None
    try:
        if json.loads(meta_path.read_text(encoding="utf-8")).get("fingerprint") != metadata["fingerprint"]: return None
        ds = Dataset.load_from_disk(str(dataset_dir))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return ds if meta.get("contexts", meta.get("count")) == len(ds) and "conversations" in ds.column_names else None
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError): return None

def save_records(cache_dir: Path, metadata: dict[str, Any], records) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    temp_dir, dataset_dir = cache_dir / f"{DATASET_DIR_NAME}.{os.getpid()}.tmp", cache_dir / DATASET_DIR_NAME
    dataset = records if isinstance(records, Dataset) else Dataset.from_list(records)
    dataset.save_to_disk(str(temp_dir))
    if dataset_dir.exists():
        import shutil; shutil.rmtree(dataset_dir)
    os.replace(temp_dir, dataset_dir)
    meta = dict(metadata)
    meta["contexts"] = len(dataset)
    meta["conversations"] = sum(len(row["conversations"]) for row in dataset)
    tmp = cache_dir / f"records.meta.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(meta), encoding="utf-8"); os.replace(tmp, cache_dir / "records.meta.json")

def cached_load(cache_dir: Path, metadata: dict[str, Any], build, *, verbose=True) -> Dataset:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with file_lock(cache_dir / "records.lock"):
        ds = load_records(cache_dir, metadata)
        if ds is not None:
            if verbose:
                conversations = sum(len(row["conversations"]) for row in ds)
                print(f"Reusing Hugging Face dataset cache: {cache_dir / DATASET_DIR_NAME} (contexts={len(ds)}, conversation_messages={conversations})")
            return ds
        records = build(); save_records(cache_dir, metadata, records); ds = load_records(cache_dir, metadata)
        if ds is None: raise RuntimeError(f"Failed to load dataset cache: {cache_dir}")
        if verbose:
            conversations = sum(len(row["conversations"]) for row in ds)
            print(f"Saved Hugging Face dataset cache: {cache_dir / DATASET_DIR_NAME} (contexts={len(ds)}, conversation_messages={conversations})")
        return ds
