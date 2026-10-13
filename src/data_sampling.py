"""Deterministic, bounded-memory sampling for per-dataset validation."""
from __future__ import annotations

import json
import hashlib
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import random
from collections import defaultdict
from pathlib import Path

from .data import ContextRecord, _messages_from_row, render_conversation

VALIDATION_DATASETS = {
    "squad",
    "msmarcov1.1",
    "msmarcov2.1",
    "hotpotqa(distractor)",
    "race",
}

_WORKER_TOKENIZER = None
_WORKER_CONFIG = None


def _init_filter_worker(tokenizer_name, data_config):
    global _WORKER_TOKENIZER, _WORKER_CONFIG
    if tokenizer_name is None:
        _WORKER_TOKENIZER = None
    else:
        from transformers import AutoTokenizer
        _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(
            tokenizer_name, local_files_only=True
        )
    _WORKER_CONFIG = data_config


def _filter_chunk(args):
    source, start, end, selected_datasets, excluded_ids = args
    offsets = []
    with open(source, "rb") as handle:
        if start:
            handle.seek(start - 1)
            boundary = handle.read(1) == b"\n"
        else:
            boundary = True
        handle.seek(start)
        if not boundary:
            handle.readline()
        while True:
            position = handle.tell()
            if position >= end:
                break
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            dataset = row.get("dataset")
            if not dataset:
                raise ValueError(f"{source}: byte {position} is missing dataset")
            if dataset not in selected_datasets:
                continue
            if (dataset, str(row.get("id", ""))) in excluded_ids:
                continue
            if _WORKER_TOKENIZER is None or _usable_for_training(row, _WORKER_TOKENIZER, _WORKER_CONFIG):
                offsets.append((dataset, position))
    return offsets


def _usable_for_training(row, tokenizer, data_config):
    context = str(row.get("context", "")).strip()
    if not context:
        return False
    messages = _messages_from_row(row)
    if data_config.filter_no_qa and not any(message.role == "assistant" for message in messages):
        return False
    turns = ContextRecord(context, messages).turns()
    if not turns:
        return False
    if data_config.filter_long_context:
        length = len(tokenizer(
            context, add_special_tokens=False, truncation=True,
            max_length=data_config.max_context_tokens + 1,
        ).input_ids)
        if not 0 < length <= data_config.max_context_tokens:
            return False
    questions = [
        render_conversation(tokenizer, turn.messages)
        if getattr(tokenizer, "apply_chat_template", None)
        else turn.question
        for turn in turns
    ]
    question_ids = tokenizer(
        questions, add_special_tokens=False, truncation=True,
        max_length=data_config.max_question_tokens + 1,
    ).input_ids
    answer_ids = tokenizer(
        [turn.answer for turn in turns], add_special_tokens=False, truncation=True,
        max_length=data_config.max_answer_tokens + 1,
    ).input_ids
    if any(
        not 0 < len(question) <= data_config.max_question_tokens
        or not 0 < len(answer) <= data_config.max_answer_tokens
        for question, answer in zip(question_ids, answer_ids)
    ):
        return False
    return (
        sum(map(len, question_ids)) + sum(map(len, answer_ids))
        <= data_config.max_conversation_tokens
    )


def sample_validation_by_dataset(
    source: str | Path,
    output_dir: str | Path,
    samples_per_dataset: int | dict[str, int] = 256,
    seed: int = 42,
    excluded_ids: set[tuple[str, str]] | None = None,
    tokenizer=None,
    data_config=None,
    datasets: set[str] | tuple[str, ...] | None = None,
    num_workers: int = 1,
) -> tuple[Path, dict[str, int]]:
    """Reservoir-sample each dataset and atomically write per-dataset and merged JSONL."""
    source = Path(source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_stat = source.stat()
    selected_datasets = set(datasets or VALIDATION_DATASETS)
    unknown = selected_datasets - VALIDATION_DATASETS
    if unknown:
        raise ValueError(f"unsupported validation datasets: {sorted(unknown)}")
    sample_limits = (
        {dataset: int(samples_per_dataset) for dataset in selected_datasets}
        if isinstance(samples_per_dataset, int)
        else {dataset: int(samples_per_dataset.get(dataset, 0)) for dataset in selected_datasets}
    )
    if any(limit <= 0 for limit in sample_limits.values()):
        raise ValueError(f"all validation sample limits must be positive: {sample_limits}")
    cache_key = {
        "version": 2,
        "source": str(source.resolve()),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "seed": seed,
        "samples_per_dataset": sample_limits,
        "datasets": sorted(selected_datasets),
        "excluded_ids_sha256": hashlib.sha256(
            json.dumps(sorted(excluded_ids or ())).encode()
        ).hexdigest(),
        "tokenizer": None if tokenizer is None else {
            "name": tokenizer.name_or_path,
            "vocab_sha256": hashlib.sha256(
                json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()
            ).hexdigest(),
            "chat_template": tokenizer.chat_template,
        },
        "filtering": None if tokenizer is None else {
            name: getattr(data_config, name)
            for name in (
                "filter_no_qa", "filter_long_context", "max_context_tokens",
                "max_question_tokens", "max_answer_tokens", "max_conversation_tokens",
            )
        },
    }
    manifest_path = output_dir / "manifest.json"
    merged_path = output_dir / "1qa-val.jsonl"
    if manifest_path.exists() and merged_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            counts = manifest["sampled_rows"]
            if (
                manifest.get("cache_key") == cache_key
                and set(counts) == selected_datasets
                and counts == sample_limits
                and all(
                    (output_dir / f"{dataset.replace('/', '_').replace('(', '_').replace(')', '')}.jsonl").exists()
                    for dataset in counts
                )
            ):
                print(f"validation subset reused: {sum(counts.values())} rows", flush=True)
                return merged_path, counts
        except (OSError, ValueError, KeyError, TypeError):
            pass
    rng = random.Random(seed)
    reservoirs: dict[str, list[int]] = defaultdict(list)
    seen: dict[str, int] = defaultdict(int)
    if num_workers > 1:
        size = source.stat().st_size
        workers = min(int(num_workers), max(1, os.cpu_count() or 1))
        chunk_size = max(1, (size + workers - 1) // workers)
        chunks = [
            (str(source), start, min(size, start + chunk_size), selected_datasets, excluded_ids or set())
            for start in range(0, size, chunk_size)
        ]
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_filter_worker,
            initargs=(None if tokenizer is None else tokenizer.name_or_path, data_config),
        ) as pool:
            filtered_chunks = pool.map(_filter_chunk, chunks)
        filtered_rows = (row for chunk in filtered_chunks for row in chunk)
    else:
        def rows_from_source():
            with source.open("rb") as handle:
                line_number = 0
                while True:
                    position = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    line_number += 1
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    dataset = row.get("dataset")
                    if not dataset:
                        raise ValueError(f"{source}:{line_number} is missing dataset")
                    if dataset not in selected_datasets:
                        continue
                    if excluded_ids and (dataset, str(row.get("id", ""))) in excluded_ids:
                        continue
                    if tokenizer is None or _usable_for_training(row, tokenizer, data_config):
                        yield dataset, position
        filtered_rows = rows_from_source()
    for dataset, position in filtered_rows:
        seen[dataset] += 1
        bucket = reservoirs[dataset]
        limit = sample_limits[dataset]
        if len(bucket) < limit:
            bucket.append(position)
        else:
            index = rng.randrange(seen[dataset])
            if index < limit:
                bucket[index] = position

    selected: dict[str, list[dict]] = {}
    for dataset in sorted(selected_datasets):
        if len(reservoirs[dataset]) < sample_limits[dataset]:
            raise ValueError(
                f"{dataset} has {len(reservoirs[dataset])} validation rows; "
                f"need {sample_limits[dataset]}"
            )
        rng.shuffle(reservoirs[dataset])
        with source.open("rb") as handle:
            selected[dataset] = []
            for position in reservoirs[dataset]:
                handle.seek(position)
                selected[dataset].append(json.loads(handle.readline()))

    temporary_files = []
    counts = {}
    merged_tmp = output_dir / "1qa-val.jsonl.tmp"
    temporary_files.append((merged_tmp, output_dir / "1qa-val.jsonl"))
    with merged_tmp.open("w", encoding="utf-8") as merged:
        for dataset, rows in selected.items():
            filename = dataset.replace("/", "_").replace("(", "_").replace(")", "")
            tmp = output_dir / f"{filename}.jsonl.tmp"
            final = output_dir / f"{filename}.jsonl"
            temporary_files.append((tmp, final))
            with tmp.open("w", encoding="utf-8") as per_dataset:
                for row in rows:
                    serialized = json.dumps(row, ensure_ascii=False) + "\n"
                    per_dataset.write(serialized)
                    merged.write(serialized)
            counts[dataset] = len(rows)

    for temporary, final in temporary_files:
        temporary.replace(final)
    manifest_tmp = output_dir / "manifest.json.tmp"
    manifest = {
        "source": str(source),
        "seed": seed,
        "samples_per_dataset": sample_limits,
        "available_rows": dict(sorted(seen.items())),
        "sampled_rows": counts,
        "excluded_test_ids": len(excluded_ids or ()),
        "tokenizer_filtered": tokenizer is not None,
        "cache_key": cache_key,
    }
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    manifest_tmp.replace(output_dir / "manifest.json")
    print(f"validation subset refreshed: {sum(counts.values())} rows", flush=True)
    return output_dir / "1qa-val.jsonl", counts
