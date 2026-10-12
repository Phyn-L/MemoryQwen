#!/usr/bin/env python3
"""Measure JSONL parsing and tokenizer filtering costs independently.

The benchmark follows the filtering path used by ``src.data.ConversationDataset``:
each non-empty JSONL line is decoded, its context and conversation schema are
validated, and the context is tokenized before the long-context filter is applied.
Only the timers are separate; the source file is still read once, so the result
also includes the real end-to-end cost of building an index.

Examples:

    # Quick estimate on the first 10k rows.
    python scripts/data/benchmark_jsonl_filter.py \
        --path /data/lz/contexts/ift_c1qa.jsonl \
        --model /data/lz/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/<hash> \
        --max-rows 10000

    # Full scan using the model and data paths from a config.
    python scripts/data/benchmark_jsonl_filter.py \
        --config configs/train_baseline.yaml \
        --max-context-tokens 4096 \
        --output outputs/benchmarks/c1qa_jsonl_filter.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import _messages_from_row


def _optional_config(path: str | None, machine: str | None):
    if path is None:
        return None
    from utils.config import TrainConfig

    return TrainConfig.from_file(path, machine=machine)


def _load_tokenizer(model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=True,
        local_files_only=True,
    )


def benchmark(
    path: Path,
    *,
    tokenizer=None,
    max_context_tokens: int = 4096,
    max_input_tokens: int = 40960,
    max_rows: int | None = None,
    filter_long_context: bool = True,
    filter_no_conversation: bool = True,
    tokenizer_filter_batch_size: int = 1024,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Return timing and filtering counters for one sequential JSONL scan.

    ``json_decode_seconds`` measures only ``json.loads``.  ``schema_filter_seconds``
    covers context extraction and conversation normalization.  The tokenizer timer
    covers the tokenizer call and the length check used by the dataset filter.
    ``unattributed_seconds`` is the remaining wall time (file I/O, Python loop,
    progress rendering, and timer overhead).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if max_context_tokens < 1 or max_input_tokens < 1:
        raise ValueError("context limits must be positive")
    if tokenizer_filter_batch_size < 1:
        raise ValueError("tokenizer_filter_batch_size must be positive")
    if filter_long_context and tokenizer is None:
        raise ValueError("tokenizer is required when filter_long_context is enabled")

    counters = {
        "rows_seen": 0,
        "rows_decoded": 0,
        "rows_with_context": 0,
        "rows_with_assistant": 0,
        "tokenizer_calls": 0,
        "tokenizer_batches": 0,
        "rows_rejected_long_context": 0,
        "rows_rejected_empty_context": 0,
        "rows_rejected_no_assistant": 0,
        "rows_accepted": 0,
    }
    json_decode_seconds = 0.0
    schema_filter_seconds = 0.0
    tokenizer_filter_seconds = 0.0
    pending_contexts: list[str] = []

    def flush_tokenizer_batch() -> None:
        nonlocal tokenizer_filter_seconds
        if not pending_contexts:
            return
        counters["tokenizer_calls"] += len(pending_contexts)
        counters["tokenizer_batches"] += 1
        t0 = time.perf_counter()
        encoded = tokenizer(
            pending_contexts,
            add_special_tokens=False,
            truncation=True,
            max_length=max_input_tokens + 1,
            padding=False,
        )
        input_ids = (
            encoded.input_ids
            if hasattr(encoded, "input_ids")
            else encoded["input_ids"]
        )
        if len(input_ids) != len(pending_contexts):
            raise ValueError(
                "tokenizer returned a different number of rows for batched filtering"
            )
        lengths = [len(ids) for ids in input_ids]
        tokenizer_filter_seconds += time.perf_counter() - t0
        for length in lengths:
            if not length or length > max_context_tokens:
                counters["rows_rejected_long_context"] += 1
            else:
                counters["rows_accepted"] += 1
        pending_contexts.clear()

    started = time.perf_counter()
    total_bytes = path.stat().st_size
    progress = tqdm(
        total=total_bytes if max_rows is None else None,
        desc=f"Benchmarking {path.name}",
        unit="B",
        unit_scale=True,
        disable=not show_progress,
    )
    try:
        with path.open("rb") as handle:
            while True:
                if max_rows is not None and counters["rows_seen"] >= max_rows:
                    break
                offset = handle.tell()
                line = handle.readline()
                progress.update(handle.tell() - offset)
                if not line:
                    break
                if not line.strip():
                    continue
                counters["rows_seen"] += 1

                t0 = time.perf_counter()
                row = json.loads(line)
                json_decode_seconds += time.perf_counter() - t0
                if not isinstance(row, dict):
                    raise ValueError(
                        f"{path}:{counters['rows_seen']} must be a JSON object"
                    )
                counters["rows_decoded"] += 1

                t0 = time.perf_counter()
                context = str(row.get("context", "")).strip()
                if not context:
                    counters["rows_rejected_empty_context"] += 1
                    schema_filter_seconds += time.perf_counter() - t0
                    continue
                counters["rows_with_context"] += 1
                messages = _messages_from_row(row)
                has_assistant = any(message.role == "assistant" for message in messages)
                if has_assistant:
                    counters["rows_with_assistant"] += 1
                elif filter_no_conversation:
                    counters["rows_rejected_no_assistant"] += 1
                    schema_filter_seconds += time.perf_counter() - t0
                    continue
                schema_filter_seconds += time.perf_counter() - t0

                if filter_long_context:
                    pending_contexts.append(context)
                    if len(pending_contexts) >= tokenizer_filter_batch_size:
                        flush_tokenizer_batch()
                else:
                    counters["rows_accepted"] += 1
            flush_tokenizer_batch()
    finally:
        progress.close()
    wall_seconds = time.perf_counter() - started
    measured = json_decode_seconds + schema_filter_seconds + tokenizer_filter_seconds
    result = {
        "path": str(path.resolve()),
        "file_size_bytes": total_bytes,
        "max_rows": max_rows,
        "max_context_tokens": max_context_tokens,
        "max_input_tokens": max_input_tokens,
        "tokenizer_filter_batch_size": tokenizer_filter_batch_size,
        "filter_long_context": filter_long_context,
        "filter_no_conversation": filter_no_conversation,
        **counters,
        "timing_seconds": {
            "wall": wall_seconds,
            "json_decode": json_decode_seconds,
            "schema_filter": schema_filter_seconds,
            "tokenizer_filter": tokenizer_filter_seconds,
            "unattributed": max(0.0, wall_seconds - measured),
        },
        "throughput": {
            "rows_per_second": counters["rows_seen"] / wall_seconds
            if wall_seconds else 0.0,
            "bytes_per_second": total_bytes / wall_seconds if wall_seconds else 0.0,
        },
    }
    return result


def _print_summary(result: dict[str, Any]) -> None:
    timing = result["timing_seconds"]
    wall = timing["wall"] or 1.0
    print("\nJSONL filtering benchmark")
    print(f"  path: {result['path']}")
    print(
        f"  rows: seen={result['rows_seen']:,}, "
        f"accepted={result['rows_accepted']:,}, "
        f"long={result['rows_rejected_long_context']:,}, "
        f"no_assistant={result['rows_rejected_no_assistant']:,}"
    )
    print(f"  wall:              {timing['wall']:.3f}s")
    print(
        f"  JSON decode:       {timing['json_decode']:.3f}s "
        f"({timing['json_decode'] / wall:.1%})"
    )
    print(
        f"  schema filtering:  {timing['schema_filter']:.3f}s "
        f"({timing['schema_filter'] / wall:.1%})"
    )
    print(
        f"  tokenizer filtering:{timing['tokenizer_filter']:.3f}s "
        f"({timing['tokenizer_filter'] / wall:.1%})"
    )
    print(
        f"  other/I/O:         {timing['unattributed']:.3f}s "
        f"({timing['unattributed'] / wall:.1%})"
    )
    print(
        f"  throughput:        {result['throughput']['rows_per_second']:,.1f} rows/s, "
        f"{result['throughput']['bytes_per_second'] / 1024**2:,.1f} MiB/s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional training config.")
    parser.add_argument("--machine", default=None)
    parser.add_argument("--path", type=Path, default=None)
    parser.add_argument("--model", default=None, help="Local tokenizer path.")
    parser.add_argument("--max-context-tokens", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--tokenizer-filter-batch-size", type=int, default=1024)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--no-long-context-filter", action="store_true")
    parser.add_argument("--keep-no-assistant", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = _optional_config(args.config, args.machine)
    path = args.path or (
        Path(cfg.data.conversation_file)
        if cfg is not None and cfg.data.conversation_file
        else None
    )
    if path is None:
        parser.error("--path is required unless --config supplies data.conversation_file")
    model = args.model or (cfg.model.name_or_path if cfg is not None else None)
    filter_long_context = not args.no_long_context_filter
    tokenizer = _load_tokenizer(model) if filter_long_context else None
    max_context_tokens = args.max_context_tokens
    if max_context_tokens is None:
        max_context_tokens = cfg.data.max_context_tokens if cfg is not None else 4096
    max_input_tokens = args.max_input_tokens
    if max_input_tokens is None:
        max_input_tokens = (
            cfg.data.max_input_tokens if cfg is not None else 40960
        )

    result = benchmark(
        path,
        tokenizer=tokenizer,
        max_context_tokens=max_context_tokens,
        max_input_tokens=max_input_tokens,
        max_rows=args.max_rows,
        filter_long_context=filter_long_context,
        filter_no_conversation=not args.keep_no_assistant,
        tokenizer_filter_batch_size=args.tokenizer_filter_batch_size,
        show_progress=not args.no_progress,
    )
    _print_summary(result)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(f"  wrote: {args.output}")


if __name__ == "__main__":
    main()
