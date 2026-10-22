"""Filter a raw 1QA meta split into the cache a training run actually reads.

``scripts/data/build_1qa.py`` writes raw, unfiltered splits (``meta/meta_<split>.json``: one
line per context with every QA pair). This module applies the *config's* filter rules --
``data.limits.{max_context_tokens,max_question_tokens,max_answer_tokens}``,
``filter_long_context``, ``filter_no_qa`` and the
``train_datasets``/``validation_datasets`` selection -- and writes ``cache/train.jsonl`` and
``cache/validation.jsonl`` plus ``cache/build.json`` (the fingerprint used to detect staleness).

The filter is the *same* function the loader uses (``filter_rows_batch``), so a row that
survives here is exactly a row the loader would keep: one rule set, not a build-time copy of
it. Rows keep **all** of their QA pairs; per-epoch turn sampling
(``data.max_qa_sample_per_context``)
still happens in the collate.

``scripts/data/build_cache.py`` is the CLI, and ``scripts/train.py`` calls :func:`ensure_cache`
so a run never trains on a cache that this config did not produce.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

from tqdm.auto import tqdm

from .cache_utils import compute_fingerprint, file_lock, tokenizer_fingerprint
from .data import (
    CONTEXT_LENGTHS_VERSION,
    context_lengths,
    context_lengths_path,
    filter_options,
    filter_row_chunks,
    progress_enabled,
)

CACHE_SPLITS = ("train", "validation")
DEFAULT_BATCH_SIZE = 512
#: Filtering is tokenization-bound (the loader-side filter is the same code), so the build runs
#: it in a process pool by default. Every rank of a run shares one build (file lock), so this is
#: one pool, not one per rank. The pool forks, which Python deprecates for a parent that already
#: holds threads (the HF tokenizer does); it is safe here because the workers only run the pure
#: filter and their output is verified identical to the serial path (see tests/test_build_cache).
DEFAULT_BUILD_WORKERS = max(1, min(8, os.cpu_count() or 1))
#: Cap on how much text one pool task carries: a row of long passages is large, and a 512-row
#: task would otherwise be tens of megabytes to pickle.
DEFAULT_MAX_CHUNK_BYTES = 8 * 1024 * 1024


def _iter_rows(path: Path) -> Iterator[tuple[int, dict]]:
    """Yield ``(line_bytes, row)`` so the chunker can bound a task by payload size."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield len(line), json.loads(line)


def _iter_chunks(
    path: Path, selected: set[str] | None, max_rows: int, max_bytes: int, on_progress=None,
):
    """Yield ``(rows_read, chunk)``: selected rows, bounded by row count and payload bytes.

    Reading and the dataset selection stay in the parent process (cheap); only the
    tokenization-heavy rule check goes to a worker. ``on_progress(rows_read, chunk_bytes)`` is
    called as the file is consumed (every row counts, selected or not) so the caller can drive a
    progress bar without a second pass over the file.
    """
    chunk: list[dict] = []
    read = 0
    size = 0
    consumed = 0
    for line_bytes, row in _iter_rows(path):
        read += 1
        consumed += line_bytes
        if selected is not None and _row_dataset(row) not in selected:
            continue
        chunk.append(row)
        size += line_bytes
        if len(chunk) >= max_rows or size >= max_bytes:
            if on_progress is not None:
                on_progress(read, consumed)
                consumed = 0
            yield read, chunk
            chunk, size = [], 0
    if on_progress is not None and consumed:
        on_progress(read, consumed)
    if chunk:
        yield read, chunk


def _selected_datasets(data_config, split: str) -> set[str] | None:
    """The dataset selection of one split; ``None`` means every dataset."""
    selection = data_config.train_datasets if split == "train" else data_config.validation_datasets
    if selection in (None, "all"):
        return None
    return {str(name) for name in selection}


def _row_dataset(row: dict) -> str:
    dataset = row.get("dataset")
    if not dataset:
        raise ValueError(f"meta row {row.get('id')!r} is missing its dataset field")
    return str(dataset)


def fingerprint_payload(data_config, split: str, source: Path, tokenizer) -> dict:
    """Everything that changes the filtered output, before hashing.

    Kept as a dict (and recorded in the manifest) so a stale cache can be diffed instead of
    only announcing that it is stale.
    """
    stat = source.stat()
    return {
        "version": 2,
        "split": split,
        "source": str(source.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "tokenizer": tokenizer_fingerprint(tokenizer),
        "datasets": sorted(_selected_datasets(data_config, split) or ()),
        "options": list(
            filter_options(
                data_config.max_context_tokens,
                data_config.filter_long_context,
                data_config.filter_no_qa,
                data_config.max_question_tokens,
                data_config.max_answer_tokens,
                data_config.filter_qa_pair_answers,
            )
        ),
    }


def cache_fingerprint(data_config, split: str, source: Path, tokenizer) -> str:
    """Hash of :func:`fingerprint_payload`; the staleness check compares these."""
    return compute_fingerprint(fingerprint_payload(data_config, split, source, tokenizer))


def filter_split(
    data_config, split: str, source: Path, tokenizer, batch_size: int = DEFAULT_BATCH_SIZE,
    quiet: bool = False, write: bool = True, num_workers: int | None = None,
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES, progress: bool | None = None,
    write_lengths: bool = True,
) -> dict:
    """Filter one meta split and (unless ``write=False``) write ``cache/<split>.jsonl``."""
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(
            f"raw {split} split not found: {source}\n"
            "Build it first: python scripts/data/build_1qa.py"
        )
    cache_path = Path(data_config.cache_file(split))
    selected = _selected_datasets(data_config, split)
    options = filter_options(
        data_config.max_context_tokens,
        data_config.filter_long_context,
        data_config.filter_no_qa,
        data_config.max_question_tokens,
        data_config.max_answer_tokens,
        data_config.filter_qa_pair_answers,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")

    workers = DEFAULT_BUILD_WORKERS if num_workers is None else int(num_workers)
    if workers <= 1:
        workers = 0
    show_progress = (not quiet) if progress is None else (bool(progress) and not quiet)
    scanned = kept = pairs = 0
    handle = temporary.open("w", encoding="utf-8") if write else None
    bar = (
        tqdm(
            total=source.stat().st_size,
            desc=f"Filtering {source.name}",
            unit="B",
            unit_scale=True,
            leave=True,
        )
        if show_progress and progress_enabled()
        else None
    )
    # `rows_read` follows the parent's reading (it is what "scanned" means), so it is taken
    # from the generator as it produces chunks, not from the workers' completion order.
    rows_read = 0

    lengths: list[int] = []

    def flush(accepted, chunk_lengths=None) -> None:
        nonlocal kept, pairs
        for row in accepted:
            if handle is not None:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            pairs += len(row.get("qa_pairs") or ())
        if chunk_lengths:
            lengths.extend(chunk_lengths)

    def report(read: int, chunk_bytes: int) -> None:
        if bar is not None:
            # `total` is the file size, so a fully consumed file reaches 100% even when rows
            # are skipped by the dataset selection.
            bar.update(chunk_bytes)
            bar.set_postfix(rows=read, contexts=kept, qa_pairs=pairs, refresh=False)

    def chunks():
        nonlocal rows_read
        for read, chunk in _iter_chunks(
            source, selected, max(32, batch_size), max_chunk_bytes, on_progress=report
        ):
            rows_read = read
            yield chunk

    try:
        for accepted, chunk_lengths in filter_row_chunks(
            chunks(), tokenizer, options, workers, with_lengths=write_lengths
        ):
            flush(accepted, chunk_lengths)
        scanned = rows_read
    finally:
        if bar is not None:
            bar.set_postfix(rows=scanned, contexts=kept, qa_pairs=pairs)
            bar.close()
        if handle is not None:
            handle.close()
    if write:
        os.replace(temporary, cache_path)
        if write_lengths:
            write_context_lengths(cache_path, tokenizer, data_config, lengths, quiet=quiet)
    elif temporary.exists():
        temporary.unlink()

    info = {
        "split": split,
        "source": str(source),
        "file": str(cache_path),
        "rows_scanned": scanned,
        "contexts": kept,
        "qa_pairs": pairs,
        "fingerprint": cache_fingerprint(data_config, split, source, tokenizer),
        "fingerprint_payload": fingerprint_payload(data_config, split, source, tokenizer),
    }
    if not quiet:
        print(
            f"[cache] {split:10s} scanned={scanned:7d} contexts={kept:7d} "
            f"qa_pairs={pairs:8d} -> {cache_path}",
            flush=True,
        )
    return info


def write_manifest(data_config, infos: Iterable[dict]) -> Path:
    """Record what each cache file was built from, for the staleness check.

    Entries are merged into the existing manifest: building one split must not erase what the
    other split recorded (that would make it look stale on the next run).
    """
    manifest_path = Path(data_config.cache_file("train")).parent / "build.json"
    splits: dict[str, dict] = {}
    if manifest_path.is_file():
        try:
            splits = dict(json.loads(manifest_path.read_text(encoding="utf-8")).get("splits", {}))
        except (json.JSONDecodeError, TypeError, AttributeError):
            splits = {}
    for info in infos:
        splits[info["split"]] = info
    payload = {"built_by": "scripts/data/build_cache.py", "splits": splits}
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path


def write_context_lengths(cache_path, tokenizer, data_config, lengths, quiet=False):
    """Record the context token lengths next to a cache file; returns the file or ``None``.

    The filter measures these anyway while applying ``max_context_tokens`` (an accepted row is
    one whose truncated count fits the cap, so the count *is* the real length), which is why the
    sortish sampler can skip its own pass over every context.
    """
    if not lengths or any(value <= 0 for value in lengths):
        if not quiet:
            print("[cache] context lengths not recorded (filter_long_context is off)", flush=True)
        return None
    stat = Path(cache_path).stat()
    payload = {
        "version": CONTEXT_LENGTHS_VERSION,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "tokenizer": tokenizer_fingerprint(tokenizer),
        "max_context_tokens": data_config.max_context_tokens,
        "lengths": lengths,
    }
    path = context_lengths_path(cache_path)
    path.write_text(json.dumps(payload), encoding="utf-8")
    if not quiet:
        print(f"[cache] recorded {len(lengths)} context lengths -> {path}", flush=True)
    return path


def lengths_are_fresh(data_config, tokenizer, split: str) -> bool:
    """True when ``cache/<split>.jsonl`` already has a usable lengths sidecar."""
    from .data import read_context_lengths

    cache_path = Path(data_config.cache_file(split))
    if not cache_path.is_file() or not context_lengths_path(cache_path).is_file():
        return False
    dataset = _LengthProbe(cache_path, data_config)
    return read_context_lengths(dataset, tokenizer) is not None


class _LengthProbe:
    """The two attributes :func:`read_context_lengths` inspects (no rows are read)."""

    def __init__(self, source_path: Path, data_config):
        self.source_path = source_path
        self.cache_metadata = {"max_context_tokens": data_config.max_context_tokens}

    def __len__(self):
        # Only used to check that the sidecar covers the file; the count itself is not read,
        # so any value >= the recorded length is fine.
        return 1


def write_lengths_for_cache(
    data_config, tokenizer, split: str, quiet: bool = False, batch_size: int = 1024,
) -> Path | None:
    """Backfill the lengths sidecar for an already built cache, without re-filtering it.

    Needed once for a cache that predates this feature (or when ``filter_long_context`` was off
    during the build): it costs exactly the pass it removes, but only once, and the *filter*
    path records the lengths for free from then on.
    """
    if not data_config.filter_long_context:
        return None
    cache_path = Path(data_config.cache_file(split))
    if not cache_path.is_file():
        return None
    bar = (
        tqdm(desc=f"Measuring {cache_path.name}", unit="context")
        if not quiet and progress_enabled()
        else None
    )
    lengths: list[int] = []
    batch: list[str] = []
    try:
        for _, row in _iter_rows(cache_path):
            batch.append(str(row.get("context", "")))
            if len(batch) >= batch_size:
                lengths.extend(context_lengths(batch, tokenizer, on_batch=_bar_update(bar)))
                batch = []
        if batch:
            lengths.extend(context_lengths(batch, tokenizer, on_batch=_bar_update(bar)))
    finally:
        if bar is not None:
            bar.close()
    if not lengths:
        return None
    return write_context_lengths(cache_path, tokenizer, data_config, lengths, quiet=quiet)


def _bar_update(bar):
    if bar is None:
        return None

    def update(count):
        bar.update(count)

    return update


def build(
    data_config, tokenizer, splits: Iterable[str] = CACHE_SPLITS, quiet: bool = False,
    dry_run: bool = False, num_workers: int | None = None, progress: bool | None = None,
    write_lengths: bool = True,
) -> dict:
    """Filter the requested splits (in order) and write the cache manifest."""
    infos = []
    for split in splits:
        if split not in CACHE_SPLITS:
            raise ValueError(f"only {CACHE_SPLITS} have a filtered cache (asked {split!r})")
        source = Path(data_config.meta_file(split))
        info = filter_split(
            data_config, split, source, tokenizer, quiet=quiet, write=not dry_run,
            num_workers=num_workers, progress=progress, write_lengths=write_lengths,
        )
        infos.append(info)
    if not dry_run:
        write_manifest(data_config, infos)
    return {"splits": {info["split"]: info for info in infos}}


def cache_is_fresh(data_config, split: str, tokenizer) -> bool:
    """True when ``cache/<split>.jsonl`` was built from this config and the current meta file."""
    cache_path = Path(data_config.cache_file(split))
    source = Path(data_config.meta_file(split))
    manifest_path = cache_path.parent / "build.json"
    if not cache_path.is_file() or not source.is_file() or not manifest_path.is_file():
        return False
    try:
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))["splits"][split]
    except (json.JSONDecodeError, KeyError, TypeError):
        return False
    return recorded.get("fingerprint") == cache_fingerprint(
        data_config, split, source, tokenizer
    )


def ensure_cache(data_config, tokenizer, splits: Iterable[str] = CACHE_SPLITS,
                 quiet: bool = False, num_workers: int | None = None,
                 progress: bool | None = None) -> list[str]:
    """Build the splits whose cache is missing or stale; returns the splits that were built.

    Safe to call from every rank of a distributed run: the write is guarded by a file lock and
    a second caller re-checks freshness after taking it, so the work happens once.
    """
    requested = list(splits)
    cache_dir = Path(data_config.cache_file("train")).parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    # A fresh cache with a missing lengths sidecar is not rebuilt (that would re-filter GBs);
    # the lengths are measured once here instead, and every later run skips the sortish pass.
    for split in requested:
        if (
            cache_is_fresh(data_config, split, tokenizer)
            and not lengths_are_fresh(data_config, tokenizer, split)
        ):
            if not quiet:
                print(f"[cache] {split}: recording context lengths (one pass)", flush=True)
            write_lengths_for_cache(data_config, tokenizer, split, quiet=quiet)
    stale = [split for split in requested if not cache_is_fresh(data_config, split, tokenizer)]
    if not stale:
        return []
    built = []
    with file_lock(cache_dir / "build.lock"):
        # Re-check under the lock: another rank may have built it while we waited.
        stale = [split for split in stale if not cache_is_fresh(data_config, split, tokenizer)]
        for split in stale:
            if not quiet:
                print(f"[cache] {split}: cache missing or stale, filtering "
                      f"{data_config.meta_file(split)}", flush=True)
        if stale:
            build(data_config, tokenizer, splits=stale, quiet=quiet, num_workers=num_workers,
                  progress=progress)
            built = list(stale)
    return built
