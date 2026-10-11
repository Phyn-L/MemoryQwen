#!/usr/bin/env python3
"""Convert SHINE's large JSON-array conversation files to JSONL.

The converter keeps only the canonical training fields (``context`` and
``conversations``), validates every role/content pair, and never materializes the
source array in memory.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from collections import deque
from pathlib import Path

from tqdm import tqdm


def iter_array(path: Path, chunk_size: int = 8 * 1024 * 1024):
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        started = finished = False
        row_number = 0
        while not finished:
            chunk = handle.read(chunk_size)
            if chunk:
                buffer += chunk
            elif not buffer:
                break
            while True:
                buffer = buffer.lstrip()
                if not started:
                    if not buffer:
                        break
                    if buffer[0] != "[":
                        raise ValueError(f"{path} is not a top-level JSON array")
                    started = True
                    buffer = buffer[1:]
                buffer = buffer.lstrip()
                # The comma separating two array items can be the first bytes
                # of a newly read chunk after the previous item was decoded.
                if buffer.startswith(","):
                    buffer = buffer[1:].lstrip()
                if not buffer:
                    break
                if buffer[0] == "]":
                    finished = True
                    buffer = buffer[1:]
                    break
                try:
                    value, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError as error:
                    decode_error = error
                    break
                if not isinstance(value, dict):
                    raise ValueError(f"{path} contains a non-object array item")
                row_number += 1
                yield value
                buffer = buffer[end:].lstrip()
                if buffer.startswith(","):
                    buffer = buffer[1:]
                    continue
                if buffer.startswith("]"):
                    finished = True
                    buffer = buffer[1:]
                    break
                if buffer:
                    raise ValueError(f"invalid JSON array separator in {path}")
                break
            if finished:
                break
            if not chunk:
                tail = buffer.lstrip()
                if tail.startswith("]"):
                    break
                excerpt = tail[:160].replace("\n", "\\n")
                raise ValueError(
                    f"invalid or truncated JSON near array row {row_number + 1} in {path}: "
                    f"{decode_error.msg} at line {decode_error.lineno}, "
                    f"column {decode_error.colno}; remaining input starts with {excerpt!r}"
                )


def normalize_row(item):
    line_number, row = item
    context = str(row.get("context", "")).strip()
    conversations = row.get("conversations")
    if not context or not isinstance(conversations, list):
        raise ValueError(f"row {line_number} requires context and conversations")
    normalized = []
    skipped_turns = 0
    for turn_number, turn in enumerate(conversations, 1):
        if not isinstance(turn, dict):
            raise ValueError(f"row {line_number} turn {turn_number} is not an object")
        role = str(turn.get("role", "")).strip().lower()
        content = str(turn.get("content", "")).strip()
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"row {line_number} has invalid role in turn {turn_number}")
        if not content:
            skipped_turns += 1
            continue
        normalized.append({"role": role, "content": content})
    if not any(turn["role"] == "assistant" for turn in normalized):
        return None, skipped_turns, 1
    return {"context": context, "conversations": normalized}, skipped_turns, 0


def convert(source: Path, destination: Path, max_rows: int | None = None, workers: int = 32) -> int:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".incomplete")
    count = 0
    skipped_turns = skipped_rows = 0
    rows = iter(enumerate(iter_array(source), 1))
    pending = deque()
    executor = ProcessPoolExecutor(max_workers=workers)
    try:
        for _ in range(workers * 2):
            try:
                pending.append(executor.submit(normalize_row, next(rows)))
            except StopIteration:
                break
        with temporary.open("w", encoding="utf-8") as output, tqdm(
            desc="Converting", unit="rows", total=max_rows
        ) as progress:
            while pending:
                future = pending.popleft()
                record, empty_turns, no_assistant = future.result()
                skipped_turns += empty_turns
                skipped_rows += no_assistant
                progress.update(1)
                if record is not None:
                    output.write(json.dumps(
                        record, ensure_ascii=False, separators=(",", ":")
                    ) + "\n")
                    count += 1
                if max_rows is not None and count >= max_rows:
                    break
                try:
                    pending.append(executor.submit(normalize_row, next(rows)))
                except StopIteration:
                    continue
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    print(f"skipped empty turns={skipped_turns}, rows without assistant={skipped_rows}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    count = convert(args.source, args.destination, args.max_rows, args.workers)
    print(f"wrote {count} conversation rows to {args.destination}")


if __name__ == "__main__":
    main()
