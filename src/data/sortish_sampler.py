"""Length-aware batching sampler for context records."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

from torch.utils.data import Sampler
from tqdm.auto import tqdm

from .cache import file_lock, read_context_lengths, tokenizer_fingerprint
from .preprocessing import context_lengths, progress_enabled


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
            "tokenizer": tokenizer_fingerprint(tokenizer),
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
        with file_lock(path.with_suffix(".lock")):
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
