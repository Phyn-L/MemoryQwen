"""Materialized context dataset backed by filtered JSONL data."""
from __future__ import annotations

import json
import os
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .cache import compute_fingerprint, tokenizer_fingerprint
from .preprocessing import filter_options, filter_row_chunks, progress_enabled
from .records import ContextRecord, _messages_from_row


class ContextDataset(Dataset):
    """Materialized context records backed by a filtered JSONL cache."""

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
                 max_answer_tokens=64, filter_qa_pair_answers=False,
                 num_workers=0, pre_filtered: bool = False):
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
            raise ValueError(f'context loader expects JSONL: {self.source_path}')
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
            'filter_qa_pair_answers': filter_qa_pair_answers,
            'max_samples': max_samples,
            'filter_long_context': filter_long_context,
            'filter_no_conversation': filter_no_conversation,
            'tokenizer': tokenizer_fingerprint(tokenizer),
        }
        fingerprint = compute_fingerprint(self.cache_metadata)
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
                max_question_tokens, max_answer_tokens, filter_qa_pair_answers,
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
                     filter_qa_pair_answers,
                     num_workers):
        """Read the JSONL once and keep the rows that pass ``filter_rows_batch``.

        One implementation, two drivers: with ``num_workers > 1`` the chunks go through a
        process pool, otherwise the identical chunk function runs in-process. The reading,
        progress reporting and sample cap are shared, so the two modes cannot diverge.
        """
        options = filter_options(
            max_context_tokens, filter_long_context, filter_no_conversation,
            max_question_tokens, max_answer_tokens, filter_qa_pair_answers,
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
