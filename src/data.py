from __future__ import annotations

import hashlib
import json
import os
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn
from torch.utils.data import Dataset, Sampler
from tqdm.auto import tqdm

from utils.ddp import is_main_process

from . import cache_utils
from .qa_prompt import format_question

_FILTER_WORKER_TOKENIZER = None
_FILTER_WORKER_OPTIONS = None
_FILTER_WORKER_WITH_LENGTHS = False
_SORTISH_WORKER_TOKENIZER = None


def _init_filter_worker(tokenizer, options, with_lengths=False):
    global _FILTER_WORKER_TOKENIZER, _FILTER_WORKER_OPTIONS, _FILTER_WORKER_WITH_LENGTHS
    _FILTER_WORKER_TOKENIZER = tokenizer
    _FILTER_WORKER_OPTIONS = options
    _FILTER_WORKER_WITH_LENGTHS = with_lengths


def filter_rows_batch(rows, tokenizer, options, lengths_out: list[int] | None = None):
    """The one filter implementation; called in-process and inside the process pool.

    It is a module-level function of its arguments (rather than of worker globals) so the
    serial and the parallel driver run *the same* rules: the rules used to be written twice,
    once per driver, which is exactly how a filter silently drifts between them.
    """
    (
        max_context_tokens,
        filter_long_context,
        filter_no_conversation,
        max_question_tokens,
        max_answer_tokens,
    ) = options
    accepted = []
    for raw in rows:
        context = str(raw.get("context", "")).strip()
        if not context:
            continue
        messages = _messages_from_row(raw)
        if filter_no_conversation and not any(m.role == "assistant" for m in messages):
            continue
        turns = (
            ContextRecord("", messages, independent_qa=True).turns()
            if raw.get("qa_pairs")
            else ContextRecord("", messages).turns()
        )
        if not turns:
            continue
        if tokenizer is not None:
            length: int | None = None
            if filter_long_context:
                length = len(tokenizer(context, add_special_tokens=False,
                                       truncation=True, max_length=max_context_tokens + 1).input_ids)
                if not 0 < length <= max_context_tokens:
                    continue
            questions = [t.question for t in turns]
            qids = tokenizer(
                questions, add_special_tokens=False, truncation=True,
                max_length=max_question_tokens + 1,
            ).input_ids
            aids = tokenizer(
                [t.answer for t in turns], add_special_tokens=False, truncation=True,
                max_length=max_answer_tokens + 1,
            ).input_ids
            if any(
                not 0 < len(q) <= max_question_tokens
                or not 0 < len(a) <= max_answer_tokens
                for q, a in zip(qids, aids, strict=True)
            ) and not raw.get("qa_pairs"):
                continue
        accepted.append(raw)
        if lengths_out is not None:
            # Every accepted row satisfies `length <= max_context_tokens`, so the truncated
            # count *is* the context's true token length (which is what the sortish sampler
            # buckets by). ``-1`` means "not measured" (no tokenizer / filter_long_context off).
            lengths_out.append(length if length is not None else -1)
    return accepted


def _filter_row_batch_in_worker(rows):
    """Process-pool entry point: read the tokenizer/options the initializer installed."""
    if not _FILTER_WORKER_WITH_LENGTHS:
        return filter_rows_batch(rows, _FILTER_WORKER_TOKENIZER, _FILTER_WORKER_OPTIONS)
    lengths: list[int] = []
    accepted = filter_rows_batch(
        rows, _FILTER_WORKER_TOKENIZER, _FILTER_WORKER_OPTIONS, lengths
    )
    return accepted, lengths


def filter_options(
    max_context_tokens, filter_long_context, filter_no_conversation,
    max_question_tokens, max_answer_tokens,
):
    """The rule set :func:`filter_rows_batch` expects, in one place.

    Kept as a tuple so it can also cross a process-pool boundary; ``scripts/data/build_cache.py``
    uses this to apply exactly the filter the loader applies.
    """
    return (
        max_context_tokens, filter_long_context, filter_no_conversation,
        max_question_tokens, max_answer_tokens,
    )


def _init_sortish_worker(tokenizer):
    global _SORTISH_WORKER_TOKENIZER
    _SORTISH_WORKER_TOKENIZER = tokenizer


def _sortish_length_batch(contexts):
    encoded = _SORTISH_WORKER_TOKENIZER(
        contexts,
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )
    input_ids = (
        encoded.input_ids
        if hasattr(encoded, "input_ids")
        else encoded["input_ids"]
    )
    if len(input_ids) != len(contexts):
        raise ValueError(
            "tokenizer returned a different number of rows for sortish lengths"
        )
    return [len(ids) for ids in input_ids]


def context_lengths(contexts, tokenizer, num_workers: int = 0, batch_size: int = 1024,
                    on_batch=None):
    """Token counts of ``contexts``, batched and in order.

    ``num_workers > 1`` sends the batches through the same pool the sortish sampler uses. Used
    by ``SortishSampler`` and available to ``scripts/data/build_cache.py`` (which gets its
    lengths straight out of the filter instead; see :func:`filter_rows_batch`).
    """
    lengths: list[int] = []
    pending: list[str] = []

    def flush():
        if not pending:
            return
        encoded = tokenizer(pending, add_special_tokens=False, truncation=False, padding=False)
        input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
        if len(input_ids) != len(pending):
            raise ValueError(
                "tokenizer returned a different number of rows for sortish lengths"
            )
        lengths.extend(len(ids) for ids in input_ids)
        if on_batch is not None:
            on_batch(len(pending))
        pending.clear()

    if num_workers <= 1:
        for context in contexts:
            pending.append(context)
            if len(pending) >= batch_size:
                flush()
        flush()
        return lengths

    def batches():
        batch = []
        for context in contexts:
            batch.append(context)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_sortish_worker,
        initargs=(tokenizer,),
    ) as executor:
        for batch_lengths in _bounded_process_map(
            executor, _sortish_length_batch, batches(), num_workers * 2
        ):
            lengths.extend(batch_lengths)
            if on_batch is not None:
                on_batch(len(batch_lengths))
    return lengths


def _bounded_process_map(executor, function, items, max_pending):
    """Yield ordered results while keeping only a small number of tasks in flight."""
    iterator = iter(items)
    pending = deque()
    for _ in range(max_pending):
        try:
            pending.append(executor.submit(function, next(iterator)))
        except StopIteration:
            break
    while pending:
        future = pending.popleft()
        yield future.result()
        try:
            pending.append(executor.submit(function, next(iterator)))
        except StopIteration:
            continue


def filter_row_chunks(chunks, tokenizer, options, num_workers: int = 0, with_lengths=False):
    """Yield the accepted rows of each chunk, **in input order**.

    One driver for both callers of :func:`filter_rows_batch`: the dataset loader and
    ``src/cache_build.py``. Serial for ``num_workers <= 1``, otherwise a bounded process pool
    running the *same* chunk function, so the two modes cannot drift apart. Order is preserved
    (the pool map is bounded and drained in submission order), which is what keeps a parallel
    cache build byte-identical to a serial one.

    With ``with_lengths`` each item is ``(accepted_rows, context_lengths)`` instead of just the
    rows; the lengths come from the rule check the filter already ran, so they cost nothing.
    """
    if num_workers > 1:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_filter_worker,
            initargs=(tokenizer, options, with_lengths),
        ) as executor:
            yield from _bounded_process_map(
                executor, _filter_row_batch_in_worker, chunks, num_workers * 2
            )
    else:
        for chunk in chunks:
            if with_lengths:
                lengths: list[int] = []
                yield filter_rows_batch(chunk, tokenizer, options, lengths), lengths
            else:
                yield filter_rows_batch(chunk, tokenizer, options)


def progress_enabled() -> bool:
    """Only the main process draws progress bars; one shared rank check, not a second one."""
    return is_main_process()


@dataclass(frozen=True)
class ConversationMessage:
    role: str
    content: str
    answers: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConversationTurn:
    """One assistant target and the messages visible before it."""

    messages: tuple[ConversationMessage, ...]
    answer: str
    answers: tuple[str, ...] = ()
    dataset: str = ""
    context_id: str = ""

    @property
    def question(self) -> str:
        for message in reversed(self.messages):
            if message.role == "user":
                return message.content
        return ""

    @property
    def references(self) -> tuple[str, ...]:
        return self.answers or ((self.answer,) if self.answer else ())


@dataclass(frozen=True)
class ContextRecord:
    context: str
    conversations: tuple[ConversationMessage, ...]
    dataset: str = ""
    context_id: str = ""
    context_token_length: int | None = None
    task: str | None = None
    independent_qa: bool = False

    def __post_init__(self) -> None:
        normalized = []
        for item in self.conversations:
            if isinstance(item, ConversationMessage):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append(
                    ConversationMessage(
                        str(item["role"]),
                        str(item["content"]),
                        tuple(str(value) for value in item.get("answers", ()) or ()),
                    )
                )
            else:
                raise TypeError(f"unsupported conversation item: {type(item)!r}")
        object.__setattr__(self, "conversations", tuple(normalized))

    def turns(self) -> tuple[ConversationTurn, ...]:
        turns = []
        history = []
        pending_user = None
        for message in self.conversations:
            if message.role == "user":
                pending_user = message
                if self.independent_qa:
                    history = [message]
                else:
                    history.append(message)
            elif message.role == "assistant" and pending_user is not None:
                turns.append(
                    ConversationTurn(
                        tuple(history),
                        message.content,
                        message.answers,
                        self.dataset,
                        self.context_id,
                    )
                )
                if not self.independent_qa:
                    history.append(message)
                pending_user = None
            elif message.role == "system":
                history.append(message)
        return tuple(turns)


def render_conversation(
    tokenizer,
    messages: tuple[ConversationMessage, ...],
    *,
    enable_thinking: bool = False,
) -> str:
    """Render structured messages using the model's native chat template."""
    if not getattr(tokenizer, "apply_chat_template", None):
        raise ValueError(
            "conversation training requires a tokenizer with a chat template"
        )
    payload = [
        {"role": message.role, "content": message.content} for message in messages
    ]
    return tokenizer.apply_chat_template(
        payload,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            yield row


def _messages_from_row(row: dict[str, Any]) -> tuple[ConversationMessage, ...]:
    qa_pairs = row.get("qa_pairs")
    if isinstance(qa_pairs, list):
        messages = []
        for pair in qa_pairs:
            if not isinstance(pair, dict):
                continue
            question = str(pair.get("question", "")).strip()
            answers = pair.get("answers") or ()
            if isinstance(answers, str):
                answers = (answers,)
            answers = tuple(str(value).strip() for value in answers if str(value).strip())
            answer = str(pair.get("answer") or (answers[0] if answers else "")).strip()
            if not question or not answer:
                continue
            question = format_question(question, row.get("dataset", ""), pair.get("metadata") or {})
            messages.extend(
                (
                    ConversationMessage("user", question),
                    ConversationMessage("assistant", answer, answers),
                )
            )
        if messages or isinstance(qa_pairs, list):
            return tuple(messages)
    raw = row.get("conversations")
    if not isinstance(raw, list):
        raise ValueError(
            "conversation data rows must contain a list named 'conversations'"
        )
    messages = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"conversation item {index} must be an object")
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported conversation role: {role!r}")
        if not content:
            continue
        answers = item.get("answers", ())
        if isinstance(answers, str):
            answers = (answers,)
        messages.append(
            ConversationMessage(
                role,
                content,
                tuple(str(value).strip() for value in answers if str(value).strip()),
            )
        )
    # Preserve row-level answer references for a single-turn conversation.
    if len([message for message in messages if message.role == "assistant"]) == 1:
        references = row.get("answers") or ()
        if isinstance(references, str):
            references = (references,)
        references = tuple(str(value).strip() for value in references if str(value).strip())
        for index, message in enumerate(messages):
            if message.role == "assistant":
                messages[index] = ConversationMessage(
                    message.role, message.content, message.answers or references
                )
                break
    return tuple(messages)


#: Sidecar ``scripts/data/build_cache.py`` writes: the context token lengths of the filtered
#: rows, for free (the filter already measured them). Versioned so a stale file is ignored.
CONTEXT_LENGTHS_VERSION = 1


def context_lengths_path(source) -> Path:
    """``train.jsonl`` -> ``train.jsonl.lengths.json``."""
    return Path(str(source) + ".lengths.json")


def read_context_lengths(dataset, tokenizer):
    """The recorded context lengths for this dataset, or ``None`` when it has to be measured.

    Validated against the file it describes (size + mtime), the tokenizer and the context limit,
    and required to cover every row of the dataset; anything else falls back to measuring.
    """
    source = getattr(dataset, "source_path", None)
    if source is None:
        return None
    path = context_lengths_path(source)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != CONTEXT_LENGTHS_VERSION:
        return None
    stat = Path(source).stat()
    if payload.get("size") != stat.st_size or payload.get("mtime_ns") != stat.st_mtime_ns:
        return None
    if payload.get("tokenizer") != cache_utils.tokenizer_fingerprint(tokenizer):
        return None
    lengths = payload.get("lengths")
    if not isinstance(lengths, list) or not lengths or len(lengths) < len(dataset):
        return None
    if any(not isinstance(value, int) or value <= 0 for value in lengths):
        return None
    metadata = getattr(dataset, "cache_metadata", {}) or {}
    limit = metadata.get("max_context_tokens")
    if limit is not None and payload.get("max_context_tokens") != limit:
        return None
    return lengths[: len(dataset)]


class ConversationDataset(Dataset):
    """Materialized conversation dataset backed by a filtered JSONL cache."""

    CACHE_VERSION = 5
    #: Files at least this large get a "Reading <file>" progress bar. The rows are parsed in
    #: this process before training starts, which for the built cache is ~18 s on a 1.6 GB
    #: train.jsonl; without the bar that gap looks like a hang between the `[data]` line and
    #: the first training step.
    READ_BAR_MIN_BYTES = 64 * 1024 * 1024
    #: How often the bar refreshes its row count (set_postfix builds a dict).
    READ_BAR_POSTFIX_EVERY = 4096

    def __init__(self, path: str | Path, tokenizer=None, max_context_tokens=2048,
                 max_input_tokens=40960, max_samples=None, filter_long_context=True,
                 filter_no_conversation=True, cache_dir: str | Path | None = None,
                 allow_empty=False, tokenizer_filter_batch_size=2048,
                 max_question_tokens=96,
                 max_answer_tokens=64, num_workers=0, pre_filtered: bool = False):
        """``pre_filtered`` says the file was already built with *these* filter options.

        It is set by ``pipeline.make_context_dataset`` for ``cache/train.jsonl`` /
        ``cache/validation.jsonl`` when ``scripts/data/build_cache.py``'s fingerprint for this
        config still matches (see ``pipeline.uses_filtered_cache``). Without it the loader
        re-tokenizes every row of a file that was produced by that same filter and writes a
        second copy under ``data_cache_dir()`` -- minutes of I/O and a duplicate ``.jsonl``.
        """
        self.source_path = Path(path)
        self.pre_filtered = bool(pre_filtered)
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        # `.json` is accepted because the raw meta splits (`meta_<split>.json`) are
        # line-delimited too; the check is there to catch a wrong file, not an extension.
        if self.source_path.suffix not in {'.jsonl', '.json'}:
            raise ValueError(f'conversation loader expects JSONL: {self.source_path}')
        if not self.source_path.exists():
            raise FileNotFoundError(self.source_path)
        self.cache_metadata = {
            'version': self.CACHE_VERSION,
            'path': str(self.source_path.resolve()),
            'size': self.source_path.stat().st_size,
            'mtime_ns': self.source_path.stat().st_mtime_ns,
            'max_context_tokens': max_context_tokens,
            'max_input_tokens': max_input_tokens,
            'max_question_tokens': max_question_tokens,
            'max_answer_tokens': max_answer_tokens,
            'max_samples': max_samples,
            'filter_long_context': filter_long_context,
            'filter_no_conversation': filter_no_conversation,
            'tokenizer': cache_utils.tokenizer_fingerprint(tokenizer),
        }
        fingerprint = cache_utils.compute_fingerprint(self.cache_metadata)
        cache_path = (Path(cache_dir) / f'conversation_filtered-{fingerprint}.jsonl'
                      if cache_dir else None)
        if self.pre_filtered:
            # Already filtered by the same rules (the cache fingerprint is a hash of exactly
            # those rules plus the source file and tokenizer), so only the sample cap is left.
            self._rows = self._read_rows(self.source_path, max_samples)
        elif cache_path and cache_path.exists():
            self._rows = self._read_rows(cache_path, max_samples)
        else:
            self._rows = self._filter_rows(
                tokenizer, max_context_tokens, max_samples, filter_long_context,
                filter_no_conversation, tokenizer_filter_batch_size,
                max_question_tokens, max_answer_tokens,
                num_workers)
            if cache_path:
                self._write_rows(cache_path, self._rows)
                if progress_enabled():
                    print(f'Saved filtered conversation data: {cache_path} '
                          f'(contexts={len(self._rows)})', flush=True)
        if not self._rows and not allow_empty:
            raise RuntimeError(f'No usable conversation rows found in {self.source_path}')

    @classmethod
    def _read_rows(cls, path, limit=None):
        """Read a JSONL file, stopping after ``limit`` rows.

        The limit matters for a capped run over the full cache (``--max-samples``): reading a
        1.6 GB file to keep 20k rows would be minutes of work for nothing. A large uncapped
        read shows a bar, because it happens before the first training step.
        """
        rows = []
        size = path.stat().st_size
        progress = (
            Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TextColumn("[green]{task.fields[rows]}"),
                console=Console(stderr=True),
                transient=False,
                refresh_per_second=5,
            )
            if (
                limit is None
                and size >= cls.READ_BAR_MIN_BYTES
                and progress_enabled()
            )
            else None
        )
        task = None
        if progress is not None:
            task = progress.add_task(f"Reading {path.name}", total=size, rows="0 rows")
            progress.start()
        try:
            with path.open(encoding='utf-8') as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    rows.append(json.loads(line))
                    if progress is not None:
                        progress.update(
                            task,
                            advance=len(line.encode('utf-8')),
                            refresh=False,
                        )
                        if len(rows) % cls.READ_BAR_POSTFIX_EVERY == 0:
                            progress.update(task, rows=f"{len(rows):,} rows", refresh=False)
                    if limit is not None and len(rows) >= limit:
                        break
        finally:
            if progress is not None:
                progress.update(task, rows=f"{len(rows):,} rows")
                progress.stop()
        return rows

    @staticmethod
    def _write_rows(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
        with temporary.open('w', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        os.replace(temporary, path)

    def _filter_rows(self, tokenizer, max_context_tokens, max_samples,
                     filter_long_context, filter_no_conversation, batch_size,
                     max_question_tokens, max_answer_tokens,
                     num_workers):
        """Read the JSONL once and keep the rows that pass ``filter_rows_batch``.

        One implementation, two drivers: with ``num_workers > 1`` the chunks go through a
        process pool, otherwise the identical chunk function runs in-process. The reading,
        progress reporting and sample cap are shared, so the two modes cannot diverge.
        """
        options = filter_options(
            max_context_tokens, filter_long_context, filter_no_conversation,
            max_question_tokens, max_answer_tokens,
        )
        chunk_size = max(32, batch_size)
        rows = []
        progress = tqdm(
            total=max_samples,
            desc=f"Filtering {self.source_path.name}",
            unit="context",
            disable=not progress_enabled(),
        )
        scan_progress = tqdm(
            total=self.source_path.stat().st_size,
            desc=f"Reading {self.source_path.name}",
            unit="B",
            unit_scale=True,
            disable=not progress_enabled(),
        )

        def iter_chunks():
            with self.source_path.open(encoding="utf-8") as handle:
                chunk = []
                for line in handle:
                    scan_progress.update(len(line.encode("utf-8")))
                    if not line.strip():
                        continue
                    chunk.append(json.loads(line))
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
                if chunk:
                    yield chunk

        def keep(accepted):
            rows.extend(accepted)
            progress.update(len(accepted))
            return max_samples is not None and len(rows) >= max_samples

        try:
            for accepted in filter_row_chunks(
                iter_chunks(), tokenizer, options, num_workers
            ):
                if keep(accepted):
                    return rows[:max_samples]
        finally:
            progress.close()
            scan_progress.close()
        return rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        row = self._rows[index]
        return ContextRecord(
            str(row["context"]).strip(),
            _messages_from_row(row),
            str(row.get("dataset", self.source_path.stem)),
            str(row.get("context_id", row.get("id", ""))),
            None,
            None,
            bool(row.get("qa_pairs")),
        )


class SortishSampler(Sampler[int]):
    CACHE_VERSION = 5

    def __init__(
        self,
        dataset,
        tokenizer,
        batch_size,
        bucket_multiplier=50,
        seed=42,
        chat_template_enable_thinking=False,
        cache_dir=None,
        num_workers=0,
    ):
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        self.dataset, self.batch_size, self.seed, self.epoch = (
            dataset,
            batch_size,
            seed,
            0,
        )
        self.start_batch = 0
        metadata = self._cache_metadata(tokenizer, chat_template_enable_thinking)
        self.lengths = self._load_or_build_lengths(
            tokenizer,
            chat_template_enable_thinking,
            Path(cache_dir) if cache_dir else None,
            metadata,
            num_workers,
        )
        self.bucket_multiplier = bucket_multiplier
        self._rebuild()

    def _cache_metadata(self, tokenizer, enable_thinking):
        payload = {
            "version": self.CACHE_VERSION,
            "dataset": getattr(self.dataset, "cache_metadata", {}),
            "record_count": len(self.dataset),
            "tokenizer": cache_utils.tokenizer_fingerprint(tokenizer),
            "chat_template_enable_thinking": enable_thinking,
        }
        payload["fingerprint"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        return payload

    def _build_lengths(self, tokenizer, enable_thinking, num_workers=0):
        progress = tqdm(
            total=len(self.dataset),
            desc="Preparing sortish lengths",
            unit="context",
            disable=not progress_enabled(),
        )
        try:
            return context_lengths(
                (record.context for record in self.dataset),
                tokenizer,
                num_workers,
                batch_size=max(
                    1, int(getattr(self.dataset, "tokenizer_filter_batch_size", 1024))
                ),
                on_batch=progress.update,
            )
        finally:
            progress.close()

    def _load_or_build_lengths(
        self, tokenizer, enable_thinking, cache_dir, metadata, num_workers=0
    ):
        recorded = read_context_lengths(self.dataset, tokenizer)
        if recorded is not None:
            if progress_enabled():
                print(
                    f"[data] reusing {len(recorded)} context lengths recorded next to "
                    f"{getattr(self.dataset, 'source_path', '')} (no sortish pass)",
                    flush=True,
                )
            return recorded
        if cache_dir is None:
            return self._build_lengths(tokenizer, enable_thinking, num_workers)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"sortish_lengths-{metadata['fingerprint']}.json"
        with cache_utils.file_lock(path.with_suffix(".lock")):
            if path.exists():
                try:
                    cached = json.loads(path.read_text())
                    lengths = cached.get("lengths")
                    if (
                        cached.get("metadata", {}).get("fingerprint")
                        == metadata["fingerprint"]
                        and isinstance(lengths, list)
                        and len(lengths) == len(self.dataset)
                    ):
                        return lengths
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    pass
            lengths = self._build_lengths(tokenizer, enable_thinking, num_workers)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"metadata": metadata, "lengths": lengths}))
            os.replace(temporary, path)
            return lengths

    def _rebuild(self):
        indices = range(len(self.lengths))
        size = max(self.batch_size, self.batch_size * self.bucket_multiplier)
        order = sorted(indices, key=self.lengths.__getitem__)
        self.buckets = []
        for i in range(0, len(order), size):
            bucket = order[i : i + size]
            random.Random(self.seed + i + self.epoch).shuffle(bucket)
            self.buckets.extend(
                bucket[j : j + self.batch_size]
                for j in range(0, len(bucket), self.batch_size)
            )

    def __iter__(self):
        batches = list(self.buckets)
        random.Random(self.seed + self.epoch).shuffle(batches)
        indices = [i for batch in batches for i in batch]
        return iter(indices[self.start_batch * self.batch_size:])

    def __len__(self):
        return max(0, sum(len(batch) for batch in self.buckets) - self.start_batch * self.batch_size)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._rebuild()

    def set_start_batch(self, batch: int = 0):
        """Start iteration at a batch boundary without materializing skipped samples."""
        self.start_batch = max(0, int(batch))


def _encode(tokenizer, texts, max_length, padding_side=None):
    previous = tokenizer.padding_side
    if padding_side is not None:
        tokenizer.padding_side = padding_side
    try:
        return tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
    finally:
        tokenizer.padding_side = previous


def _user_prefix_text(text):
    """Return the rendered conversation prefix through its first user header."""
    marker = "<|im_start|>user\n"
    boundary = text.find(marker)
    if boundary < 0:
        return ""
    boundary += len(marker)
    return text[:boundary]


def split_chat_prompt(text):
    """Split memory insertion prefix, complete chat history, and generation header."""
    prefix = _user_prefix_text(text)
    boundary = text.rfind("<|im_end|>")
    if boundary < 0:
        return prefix, text[len(prefix):], ""
    return prefix, text[len(prefix):boundary], text[boundary:]


def _trim_prompt_parts(prefix_ids, suffix_ids, max_length, tokenizer):
    """Cap a single-turn prompt while retaining the user header and latest question text."""
    if len(prefix_ids) + len(suffix_ids) <= max_length:
        return prefix_ids, suffix_ids
    header_ids = tokenizer("<|im_start|>user\n", add_special_tokens=False).input_ids
    if not prefix_ids:
        return [], suffix_ids[-max_length:]
    if not header_ids or prefix_ids[-len(header_ids):] != header_ids:
        raise ValueError("tokenized QA prefix does not end at the user header")
    suffix_budget = max(0, max_length - len(header_ids))
    suffix_ids = suffix_ids[-suffix_budget:] if suffix_budget else []
    prefix_budget = max_length - len(suffix_ids)
    return prefix_ids[-prefix_budget:], suffix_ids


def _turn_rng(record, epoch):
    """Deterministic RNG for the QA turns of one context in one epoch.

    Keyed on the record (dataset + id + context text) and the epoch, so which QA turns a
    context contributes no longer depends on which DataLoader worker happened to pick it up,
    nor on how many workers there are. ``random.sample`` on the process-global stream did
    depend on both: changing ``data.num_workers`` silently changed the training distribution,
    and a resumed run could not reproduce the rows it had already consumed.
    """
    key = "\x00".join(
        (str(record.dataset), str(record.context_id), record.context, str(int(epoch)))
    ).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.blake2s(key, digest_size=8).digest(), "big"))


def _sample_turns(record, turns, count, epoch):
    """Pick ``count`` turns of one context, deterministically (a permutation when it is all)."""
    count = min(int(count), len(turns))
    if count <= 0 or not turns:
        return []
    return _turn_rng(record, epoch).sample(turns, count)


def collate_context_records(
    rows,
    tokenizer,
    max_context_tokens=2048,
    max_question_tokens=96,
    max_answer_tokens=64,
    append_eos=True,
    chat_template_enable_thinking=False,
    max_qa_sample_per_context=4,
    sample_qa=True,
    question_padding_side="left",
    training=False,
    train_stage="mix",
    epoch=0,
):
    """Build memory-prefix QA batches from structured conversations.

    Every selected turn is rendered as ``history + current user + generation prompt``.
    The current assistant content is kept as the target, so PWC history is visible while
    the answer being supervised is never leaked in the input.

    ``epoch`` only feeds the deterministic QA-turn sampling (:func:`_turn_rng`); the same
    records and epoch always produce the same batch, whatever the worker count.
    """
    if not getattr(tokenizer, "apply_chat_template", None):
        raise ValueError("conversation training requires tokenizer.apply_chat_template")
    if train_stage not in {"two_stage", "only_qa", "only_recon", "mix"}:
        raise ValueError("train_stage must be two_stage, only_qa, only_recon or mix")
    contexts = [row.context for row in rows]
    c = _encode(tokenizer, contexts, max_context_tokens)
    selected: list[ConversationTurn] = []
    mapping: list[int] = []
    task_ids: list[int] = []
    for row_index, record in enumerate(rows):
        turns = list(record.turns())
        if training and train_stage in {"two_stage", "only_qa", "mix"} and not turns:
            raise ValueError(
                "conversation training requires at least one assistant turn"
            )
        if training and train_stage in {"two_stage", "mix"}:
            qa_turns = _sample_turns(record, turns, max_qa_sample_per_context, epoch)
            chosen = [
                *qa_turns,
                ConversationTurn(
                    (),
                    record.context,
                    dataset=record.dataset,
                    context_id=record.context_id,
                ),
            ]
            chosen_tasks = [1] * len(qa_turns) + [0]  # 1=QA, 0=Recon
        elif training and train_stage == "only_recon":
            chosen = [
                ConversationTurn(
                    (),
                    record.context,
                    dataset=record.dataset,
                    context_id=record.context_id,
                )
            ]
            chosen_tasks = [0]
        elif training:
            chosen = _sample_turns(record, turns, max_qa_sample_per_context, epoch)
            chosen_tasks = [1] * len(chosen)
        else:
            chosen = (
                _sample_turns(record, turns, max_qa_sample_per_context, epoch)
                if sample_qa else turns
            )
            chosen_tasks = [1] * len(chosen)
        selected.extend(chosen)
        mapping.extend([row_index] * len(chosen))
        task_ids.extend(chosen_tasks)
    if not selected:
        raise ValueError("conversation batch contains no assistant turns")

    rendered = []
    for turn, task in zip(selected, task_ids, strict=True):
        if task == 0:
            rendered.append("")
            continue
        messages = turn.messages
        payload = [{"role": item.role, "content": item.content} for item in messages]
        text = tokenizer.apply_chat_template(
            payload,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=chat_template_enable_thinking,
        )
        rendered.append(text)
    # The template already contains role markers and the assistant generation marker.
    qa_rendered = [text for text, task in zip(rendered, task_ids, strict=True) if task == 1]
    qa_turns = [turn for turn, task in zip(selected, task_ids, strict=True) if task == 1]
    prefix_rows, suffix_rows, header_rows = [], [], []
    if qa_rendered:
        prefix_texts = [_user_prefix_text(text) for text in qa_rendered]
        body_texts = ["".join(split_chat_prompt(text)[:2]) for text in qa_rendered]
        all_encoded = tokenizer(qa_rendered, add_special_tokens=False, padding=True)
        prefix_encoded = tokenizer(prefix_texts, add_special_tokens=False, padding=True)
        body_encoded = tokenizer(body_texts, add_special_tokens=False, padding=True)
        all_ids = [
            [value for value, keep in zip(ids, mask, strict=True) if keep]
            for ids, mask in zip(all_encoded.input_ids, all_encoded.attention_mask, strict=True)
        ]
        encoded_prefixes = [
            [value for value, keep in zip(ids, mask, strict=True) if keep]
            for ids, mask in zip(
                prefix_encoded.input_ids, prefix_encoded.attention_mask, strict=True
            )
        ]
        encoded_bodies = [
            [value for value, keep in zip(ids, mask, strict=True) if keep]
            for ids, mask in zip(body_encoded.input_ids, body_encoded.attention_mask, strict=True)
        ]
    else:
        all_ids, encoded_prefixes, encoded_bodies = [], [], []
    for row_index, (ids, prefix, body) in enumerate(
        zip(all_ids, encoded_prefixes, encoded_bodies, strict=True)
    ):
        ids, prefix = list(ids), list(prefix)
        if ids[:len(prefix)] != prefix:
            raise ValueError("chat tokenizer does not preserve the user-header token boundary")
        if ids[:len(body)] != body:
            raise ValueError("chat tokenizer does not preserve the user-end token boundary")
        suffix = ids[len(prefix):len(body)]
        turn = qa_turns[row_index]
        single_turn_qa = len(turn.messages) <= 2 and turn.dataset != "race"
        if single_turn_qa:
            prefix, suffix = _trim_prompt_parts(
                prefix, suffix, max_question_tokens, tokenizer
            )
        prefix_rows.append(prefix)
        suffix_rows.append(suffix)
        header_rows.append(ids[len(body):])
    if qa_rendered:
        previous_padding = tokenizer.padding_side
        tokenizer.padding_side = question_padding_side
        try:
            prefix_encoded = tokenizer.pad(
                {"input_ids": prefix_rows}, padding=True, return_tensors="pt",
            )
            suffix_encoded = tokenizer.pad(
                {"input_ids": suffix_rows}, padding=True, return_tensors="pt",
            )
            header_encoded = tokenizer.pad(
                {"input_ids": header_rows}, padding=True, return_tensors="pt",
            )
        finally:
            tokenizer.padding_side = previous_padding
        q_prefix = torch.zeros(
            (len(selected), prefix_encoded.input_ids.size(1)),
            dtype=prefix_encoded.input_ids.dtype,
        )
        q_prefix_mask = torch.zeros_like(q_prefix, dtype=torch.bool)
        q = torch.zeros(
            (len(selected), suffix_encoded.input_ids.size(1)),
            dtype=suffix_encoded.input_ids.dtype,
        )
        q_mask = torch.zeros_like(q, dtype=torch.bool)
        q_header = torch.zeros(
            (len(selected), header_encoded.input_ids.size(1)), dtype=q.dtype,
        )
        q_header_mask = torch.zeros_like(q_header, dtype=torch.bool)
        qa_rows = torch.tensor([task == 1 for task in task_ids])
        q_prefix[qa_rows] = prefix_encoded.input_ids
        q_prefix_mask[qa_rows] = prefix_encoded.attention_mask.bool()
        q[qa_rows] = suffix_encoded.input_ids
        q_mask[qa_rows] = suffix_encoded.attention_mask.bool()
        q_header[qa_rows] = header_encoded.input_ids
        q_header_mask[qa_rows] = header_encoded.attention_mask.bool()
    else:
        q_prefix = torch.empty((len(selected), 0), dtype=c.input_ids.dtype)
        q_prefix_mask = torch.empty((len(selected), 0), dtype=torch.bool)
        q = torch.empty((len(selected), 0), dtype=c.input_ids.dtype)
        q_mask = torch.empty((len(selected), 0), dtype=torch.bool)
        q_header = torch.empty((len(selected), 0), dtype=c.input_ids.dtype)
        q_header_mask = torch.empty((len(selected), 0), dtype=torch.bool)

    # Every supervision target is tokenized exactly once, here. Both task views below only
    # re-pad subsets of this list -- re-tokenizing per task was pure duplicated work and a
    # second place for the truncation limits to drift.
    target_rows = []
    for turn, task in zip(selected, task_ids, strict=True):
        cap_qa_answer = task == 1 and len(turn.messages) <= 2 and turn.dataset != "race"
        target_rows.append(tokenizer(
            turn.answer,
            truncation=task == 0 or cap_qa_answer,
            max_length=max_context_tokens if task == 0 else max_answer_tokens,
            add_special_tokens=False,
        ).input_ids)

    task_tensor = torch.tensor(task_ids, dtype=torch.long)

    def pack(task_id: int):
        """One task's rows: right-padded targets, EOS appended, padding masked out of the labels.

        The padding + EOS + label rule lives here once and serves both schemas: the training
        batch (QA and Recon rows) and the evaluation batch (QA rows only). ``append_eos``
        applies to every supervision row, QA and Recon alike.
        """
        indices = torch.nonzero(task_tensor.eq(task_id), as_tuple=False).flatten()
        if indices.numel() == 0:
            return {
                "input_ids": q.new_empty((0, q.size(1) if task_id == 1 else 0)),
                "input_mask": q_mask.new_empty((0, q_mask.size(1) if task_id == 1 else 0)),
                "prefix_ids": q_prefix.new_empty((0, q_prefix.size(1) if task_id == 1 else 0)),
                "prefix_mask": q_prefix_mask.new_empty((0, q_prefix_mask.size(1) if task_id == 1 else 0)),
                "answer_prefix_ids": q_header.new_empty((0, q_header.size(1))),
                "answer_prefix_mask": q_header_mask.new_empty((0, q_header_mask.size(1))),
                "target_ids": q.new_empty((0, 0)),
                "target_mask": q_mask.new_empty((0, 0)),
                "labels": q.new_empty((0, 0)),
                "context_indices": indices,
            }
        token_lists = [target_rows[index] for index in indices.tolist()]
        previous = tokenizer.padding_side
        tokenizer.padding_side = "right"
        try:
            target = tokenizer.pad(
                {"input_ids": token_lists}, padding=True, return_tensors="pt"
            )
        finally:
            tokenizer.padding_side = previous
        if append_eos and getattr(tokenizer, "eos_token_id", None) is not None:
            eos = int(tokenizer.eos_token_id)
            ids = torch.full(
                (target.input_ids.size(0), target.input_ids.size(1) + 1),
                int(tokenizer.pad_token_id),
                dtype=target.input_ids.dtype,
            )
            mask = torch.zeros_like(ids)
            for row in range(target.input_ids.size(0)):
                valid = int(target.attention_mask[row].sum())
                ids[row, :valid] = target.input_ids[row, :valid]
                mask[row, :valid + 1] = 1
                ids[row, valid] = eos
            target.input_ids, target.attention_mask = ids, mask
        target_labels = torch.full_like(target.input_ids, -100)
        target_labels[target.attention_mask.bool()] = target.input_ids[
            target.attention_mask.bool()
        ]
        return {
            "input_ids": q[indices] if task_id == 1 else q.new_empty((indices.numel(), 0)),
            "input_mask": (
                q_mask[indices] if task_id == 1 else q_mask.new_empty((indices.numel(), 0))
            ),
            "prefix_ids": (
                q_prefix[indices] if task_id == 1 else q_prefix.new_empty((indices.numel(), 0))
            ),
            "prefix_mask": (
                q_prefix_mask[indices]
                if task_id == 1 else q_prefix_mask.new_empty((indices.numel(), 0))
            ),
            "target_ids": target.input_ids,
            "answer_prefix_ids": q_header[indices],
            "answer_prefix_mask": q_header_mask[indices],
            "target_mask": target.attention_mask.bool(),
            "labels": target_labels,
            "context_indices": torch.tensor(
                [mapping[index] for index in indices.tolist()], dtype=torch.long
            ),
        }

    qa = pack(1)
    # One batch schema for training and evaluation: the evaluator and ``scripts/test.py``
    # read the same names, and the Recon half is simply absent from an evaluation batch.
    batch = {
        "context_ids": c.input_ids,
        "context_mask": c.attention_mask.bool(),
        "qa_input_ids": qa["input_ids"],
        "qa_input_mask": qa["input_mask"],
        "qa_prefix_ids": qa["prefix_ids"],
        "qa_prefix_mask": qa["prefix_mask"],
        "qa_answer_prefix_ids": qa["answer_prefix_ids"],
        "qa_answer_prefix_mask": qa["answer_prefix_mask"],
        "qa_target_ids": qa["target_ids"],
        "qa_target_mask": qa["target_mask"],
        "qa_labels": qa["labels"],
        "qa_context_indices": qa["context_indices"],
        "records": selected,
    }
    if training:
        recon = pack(0)
        batch.update(
            {
                "recon_input_ids": recon["input_ids"],
                "recon_input_mask": recon["input_mask"],
                "recon_target_ids": recon["target_ids"],
                "recon_target_mask": recon["target_mask"],
                "recon_labels": recon["labels"],
                "recon_context_indices": recon["context_indices"],
            }
        )
    return batch


collate_fn = collate_context_records
