from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler
from tqdm.auto import tqdm

from utils.ddp import is_main_process

from . import dataset_cache


def _progress_enabled() -> bool:
    """Only the main process draws progress bars; one shared rank check, not a second one."""
    return is_main_process()


@dataclass(frozen=True)
class QARecord:
    question: str
    answer: str
    dataset: str = ""
    context_id: str = ""


@dataclass(frozen=True)
class ContextRecord:
    context: str
    qa_pairs: tuple[QARecord, ...]
    dataset: str = ""
    context_id: str = ""
    context_token_length: int | None = None


def render_question(tokenizer, question: str, use_chat_template: bool = False, enable_thinking: bool = False) -> str:
    if not use_chat_template or not getattr(tokenizer, "apply_chat_template", None):
        return question
    messages = [{"role": "user", "content": question}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _answer(qa: dict[str, Any]) -> str | None:
    values = qa.get("answers", qa.get("answer", []))
    values = values if isinstance(values, list) else [values]
    return next((str(value).strip() for value in values if value is not None and str(value).strip()), None)


class AggregatedContextDataset(Dataset):
    """Context-level dataset: one source JSONL row becomes one item."""

    def __init__(self, root, dataset="all", split="train", tokenizer=None, max_context_tokens=2048, max_samples=None, filter_long_context=True, filter_no_qa=True, allow_empty=False, cache_dir=None):
        root = Path(root)
        names = [p.name for p in sorted(root.iterdir()) if p.is_dir()] if dataset in (None, "all") else ([dataset] if isinstance(dataset, str) else list(dataset))
        files = [(name, root / name / f"{split}.jsonl") for name in names]
        files = [(name, path) for name, path in files if path.exists()]
        metadata = dataset_cache.build_metadata(files, split=split, max_context_tokens=max_context_tokens, filter_long_context=filter_long_context, filter_no_qa=filter_no_qa)
        self.cache_metadata = {"root": str(root.resolve()), "datasets": names, "max_samples": max_samples, "filter": metadata}
        split_dir = Path(cache_dir) / dataset_cache.CACHE_DIR_NAME / f"{split}-{metadata['fingerprint'][:16]}" if cache_dir is not None else None
        if split_dir is not None:
            rows = dataset_cache.cached_load(split_dir, metadata, lambda: self._build_records(files, tokenizer, max_context_tokens, filter_long_context, filter_no_qa), verbose=_progress_enabled())
        else:
            rows = self._build_records(files, tokenizer, max_context_tokens, filter_long_context, filter_no_qa)
        if max_samples is not None:
            rows = rows.select(range(min(max_samples, len(rows))))
        self.dataset = rows
        if not len(rows) and not allow_empty:
            raise RuntimeError(f"No usable contexts found under {root} split={split} datasets={names}")

    def _build_records(self, files, tokenizer, max_context_tokens, filter_long_context, filter_no_qa):
        records = []
        empty_context = dropped_long = dropped_no_qa = 0
        with tqdm(files, total=len(files), desc="Loading data", unit="file", disable=not _progress_enabled()) as file_bar:
            for name, path in file_bar:
                for line in path.open(encoding="utf-8"):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    context = str(row.get("context", "")).strip()
                    if not context:
                        empty_context += 1
                        continue
                    if filter_long_context and tokenizer is not None:
                        ids = tokenizer(context, add_special_tokens=False, truncation=True, max_length=max_context_tokens + 1).input_ids
                        if len(ids) > max_context_tokens:
                            dropped_long += 1
                            continue
                    pairs = []
                    for qa in row.get("qa_pairs", []) or []:
                        answer = _answer(qa); question = str(qa.get("question", "")).strip()
                        if filter_no_qa and (not question or not answer):
                            continue
                        if question and answer:
                            pairs.append(QARecord(question, answer, name, str(row.get("context_id", ""))))
                    if pairs:
                        records.append(ContextRecord(context, tuple(pairs), name, str(row.get("context_id", ""))))
                    else:
                        dropped_no_qa += 1
                file_bar.set_postfix(contexts=len(records), qa_pairs=sum(len(r.qa_pairs) for r in records))
        # The filters drop rows silently otherwise, which makes a shrinking dataset look
        # like a data problem. Report the counts so a change of max_context_tokens or
        # dataset mix is visible in the log.
        if _progress_enabled():
            print(
                f"Loaded {len(records)} contexts from {len(files)} file(s) "
                f"(dropped: {dropped_long} longer than {max_context_tokens} tokens, "
                f"{dropped_no_qa} without a usable QA pair, {empty_context} with an empty context)"
            )
        return dataset_cache.Dataset.from_list([
            {"context": r.context, "qa_pairs": [asdict(q) for q in r.qa_pairs], "dataset": r.dataset, "context_id": r.context_id}
            for r in records
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        pairs = tuple(QARecord(str(q["question"]), str(q["answer"]), str(q.get("dataset", row.get("dataset", ""))), str(q.get("context_id", row.get("context_id", "")))) for q in row["qa_pairs"])
        return ContextRecord(str(row["context"]), pairs, str(row.get("dataset", "")), str(row.get("context_id", "")))


class SortishSampler(Sampler[int]):
    CACHE_VERSION = 3

    def __init__(self, dataset, tokenizer, batch_size, bucket_multiplier=50, seed=42, use_chat_template=False, chat_template_enable_thinking=False, cache_dir=None):
        self.dataset, self.batch_size, self.seed, self.epoch = dataset, batch_size, seed, 0
        metadata = self._cache_metadata(tokenizer, use_chat_template, chat_template_enable_thinking)
        self.lengths = self._load_or_build_lengths(tokenizer, use_chat_template, chat_template_enable_thinking, Path(cache_dir) if cache_dir else None, metadata)
        self.bucket_multiplier = bucket_multiplier; self._rebuild()

    def _cache_metadata(self, tokenizer, use_chat_template, enable_thinking):
        payload = {"version": self.CACHE_VERSION, "dataset": getattr(self.dataset, "cache_metadata", {}), "record_count": len(self.dataset), "tokenizer": str(getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)), "tokenizer_class": tokenizer.__class__.__name__, "vocab_size": getattr(tokenizer, "vocab_size", None), "use_chat_template": use_chat_template, "chat_template_enable_thinking": enable_thinking}
        payload["fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(); return payload

    def _build_lengths(self, tokenizer, use_chat_template, enable_thinking):
        lengths = []
        for record in tqdm(self.dataset, total=len(self.dataset), desc="Preparing sortish lengths", unit="context", disable=not _progress_enabled()):
            lengths.append(len(tokenizer(record.context, add_special_tokens=False).input_ids))
        return lengths

    def _load_or_build_lengths(self, tokenizer, use_chat_template, enable_thinking, cache_dir, metadata):
        if cache_dir is None: return self._build_lengths(tokenizer, use_chat_template, enable_thinking)
        cache_dir.mkdir(parents=True, exist_ok=True); path = cache_dir / "sortish_lengths.json"
        with dataset_cache.file_lock(cache_dir / "sortish_lengths.lock"):
            if path.exists():
                try:
                    cached = json.loads(path.read_text()); lengths = cached.get("lengths")
                    if cached.get("metadata", {}).get("fingerprint") == metadata["fingerprint"] and isinstance(lengths, list) and len(lengths) == len(self.dataset): return lengths
                except (OSError, ValueError, TypeError, json.JSONDecodeError): pass
            lengths = self._build_lengths(tokenizer, use_chat_template, enable_thinking); tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps({"metadata": metadata, "lengths": lengths})); os.replace(tmp, path); return lengths

    def _rebuild(self):
        size = max(self.batch_size, self.batch_size * self.bucket_multiplier); order = sorted(range(len(self.lengths)), key=self.lengths.__getitem__); self.buckets = []
        for i in range(0, len(order), size):
            bucket = order[i:i + size]; random.Random(self.seed + i + self.epoch).shuffle(bucket); self.buckets.extend(bucket[j:j + self.batch_size] for j in range(0, len(bucket), self.batch_size))

    def __iter__(self):
        batches = list(self.buckets); random.Random(self.seed + self.epoch).shuffle(batches); return iter([i for batch in batches for i in batch])
    def __len__(self): return len(self.dataset)
    def set_epoch(self, epoch): self.epoch = epoch; self._rebuild()


def _encode(tokenizer, texts, max_length, padding_side=None):
    previous = tokenizer.padding_side
    if padding_side is not None:
        tokenizer.padding_side = padding_side
    try:
        return tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length, add_special_tokens=False)
    finally:
        tokenizer.padding_side = previous


def collate_context_records(rows, tokenizer, max_context_tokens=2048, max_question_tokens=128, max_answer_tokens=128, append_eos=True, use_chat_template=False, chat_template_enable_thinking=False, qa_per_context=4, sample_qa=True, question_padding_side="right", eos_mode="overwrite"):
    contexts = [r.context for r in rows]; c = _encode(tokenizer, contexts, max_context_tokens)
    sampled, mapping = [], []
    for i, record in enumerate(rows):
        chosen = random.sample(record.qa_pairs, min(qa_per_context, len(record.qa_pairs))) if sample_qa else list(record.qa_pairs)
        sampled.extend(chosen); mapping.extend([i] * len(chosen))
    if not sampled:
        raise ValueError("context batch contains no QA pairs")
    questions = [render_question(tokenizer, q.question, use_chat_template, chat_template_enable_thinking) for q in sampled]
    q = _encode(tokenizer, questions, max_question_tokens, padding_side=question_padding_side); a = _encode(tokenizer, [x.answer for x in sampled], max_answer_tokens)
    if append_eos and getattr(tokenizer, "eos_token_id", None) is not None:
        eos = int(tokenizer.eos_token_id)
        if eos_mode == "append":
            # Always append EOS after the last real answer token. The legacy
            # "overwrite" path writes EOS over the final answer token whenever
            # that row is the longest in the batch (valid == width), destroying it.
            rows_n, width = a.input_ids.size(0), a.input_ids.size(1) + 1
            ids = torch.full((rows_n, width), int(tokenizer.pad_token_id), dtype=a.input_ids.dtype)
            mask = torch.zeros((rows_n, width), dtype=torch.long)
            for i in range(rows_n):
                valid = int(a.attention_mask[i].sum())
                ids[i, :valid] = a.input_ids[i, :valid]
                mask[i, :valid] = 1
                ids[i, valid] = eos
                mask[i, valid] = 1
            a["input_ids"] = ids
            a["attention_mask"] = mask
        else:
            for i in range(a.input_ids.size(0)):
                valid = int(a.attention_mask[i].sum())
                if valid < a.input_ids.size(1): a.input_ids[i, valid] = eos; a.attention_mask[i, valid] = True
                elif valid: a.input_ids[i, valid - 1] = eos
    labels = torch.full_like(a.input_ids, -100); labels[a.attention_mask.bool()] = a.input_ids[a.attention_mask.bool()]
    return {"context_ids": c.input_ids, "context_mask": c.attention_mask.bool(), "question_ids": q.input_ids, "question_mask": q.attention_mask.bool(), "answer_ids": a.input_ids, "answer_mask": a.attention_mask.bool(), "labels": labels, "qa_context_indices": torch.tensor(mapping, dtype=torch.long), "records": sampled}


# Public name for the collate. It used to be a pass-through wrapper that re-declared all
# twelve parameters and defaults -- which is exactly how a default silently drifts away
# from the real signature (the wrapper still said question_padding_side="right" /
# eos_mode="overwrite" long after the real defaults became "left" / "append").
collate_fn = collate_context_records
