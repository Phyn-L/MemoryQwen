from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler
from tqdm.auto import tqdm

from utils.ddp import is_main_process

from . import cache_utils


def _progress_enabled() -> bool:
    """Only the main process draws progress bars; one shared rank check, not a second one."""
    return is_main_process()


@dataclass(frozen=True)
class ConversationMessage:
    role: str
    content: str
    answers: tuple[str, ...] = ()
    answer_letter: str = ""


@dataclass(frozen=True)
class ConversationTurn:
    """One assistant target and the messages visible before it."""

    messages: tuple[ConversationMessage, ...]
    answer: str
    answers: tuple[str, ...] = ()
    answer_letter: str = ""
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
                        str(item.get("answer_letter", "")).strip().upper(),
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
                history.append(message)
            elif message.role == "assistant" and pending_user is not None:
                turns.append(
                    ConversationTurn(
                        tuple(history),
                        message.content,
                        message.answers,
                        message.answer_letter,
                        self.dataset,
                        self.context_id,
                    )
                )
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
    try:
        return tokenizer.apply_chat_template(
            payload,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(
            payload, tokenize=False, add_generation_prompt=True
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
                str(item.get("answer_letter", "")).strip().upper(),
            )
        )
    return tuple(messages)


class ConversationDataset(Dataset):
    """Random-access JSONL dataset with a compact byte-offset index."""

    CACHE_VERSION = 2

    def __init__(
        self,
        path: str | Path,
        tokenizer=None,
        max_context_tokens=2048,
        max_input_tokens=40960,
        max_samples=None,
        filter_long_context=True,
        filter_no_conversation=True,
        cache_dir: str | Path | None = None,
        allow_empty=False,
        tokenizer_filter_batch_size: int = 2048,
        max_conversation_tokens: int = 768,
        max_question_tokens: int = 96,
        max_answer_tokens: int = 64,
    ):
        self.path = Path(path)
        if self.path.suffix != ".jsonl":
            raise ValueError(f"conversation loader expects JSONL: {self.path}")
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if tokenizer_filter_batch_size <= 0:
            raise ValueError("tokenizer_filter_batch_size must be positive")
        if max_question_tokens <= 0 or max_conversation_tokens <= 0 or max_answer_tokens <= 0:
            raise ValueError("question, conversation and answer token limits must be positive")
        self.tokenizer_filter_batch_size = tokenizer_filter_batch_size
        self.max_conversation_tokens = max_conversation_tokens
        self.max_question_tokens = max_question_tokens
        self.max_answer_tokens = max_answer_tokens
        self.cache_metadata = {
            "path": str(self.path.resolve()),
            "size": self.path.stat().st_size,
            "mtime_ns": self.path.stat().st_mtime_ns,
            "max_context_tokens": max_context_tokens,
            "max_input_tokens": max_input_tokens,
            "max_conversation_tokens": max_conversation_tokens,
            "max_question_tokens": max_question_tokens,
            "max_answer_tokens": max_answer_tokens,
            "max_samples": max_samples,
            "filter_long_context": filter_long_context,
            "filter_no_conversation": filter_no_conversation,
        }
        self._offsets = self._load_or_build_offsets(
            tokenizer,
            max_context_tokens,
            max_input_tokens,
            max_samples,
            filter_long_context,
            filter_no_conversation,
            Path(cache_dir) if cache_dir else None,
        )
        if not self._offsets and not allow_empty:
            raise RuntimeError(f"No usable conversation rows found in {self.path}")
        self._handle = None

    def _cache_metadata(self, tokenizer) -> dict[str, Any]:
        metadata = {
            "version": self.CACHE_VERSION,
            **self.cache_metadata,
            "tokenizer": cache_utils.tokenizer_fingerprint(tokenizer),
        }
        metadata["fingerprint"] = cache_utils.compute_fingerprint(metadata)
        return metadata

    def _offset_cache_path(self, cache_dir: Path, metadata: dict[str, Any]) -> Path:
        return cache_dir / f"conversation_offsets-{metadata['fingerprint']}.json"

    def _load_offsets(self, path: Path, metadata: dict[str, Any]) -> list[int] | None:
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("metadata", {}).get("fingerprint") != metadata["fingerprint"]:
                return None
            offsets = cached.get("offsets")
            if not isinstance(offsets, list) or any(
                type(offset) is not int or offset < 0 for offset in offsets
            ):
                return None
            return offsets
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _save_offsets(
        self, path: Path, metadata: dict[str, Any], offsets: list[int]
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps({"metadata": metadata, "offsets": offsets}),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _build_offsets(
        self,
        tokenizer,
        max_context_tokens,
        max_input_tokens,
        max_samples,
        filter_long_context,
        filter_no_conversation,
    ) -> list[int]:
        offsets = []
        pending_offsets = []
        pending_contexts = []
        pending_questions: list[list[str]] = []
        pending_answers: list[list[str]] = []
        batch_size = self.tokenizer_filter_batch_size

        def flush_tokenizer_batch() -> bool:
            if not pending_offsets:
                return False
            if tokenizer is not None and filter_long_context:
                encoded = tokenizer(
                    pending_contexts,
                    add_special_tokens=False,
                    truncation=True,
                    # Only one token beyond the training filter is needed to
                    # distinguish <= max_context_tokens from overlong rows.
                    max_length=max_context_tokens + 1,
                    padding=False,
                )
                input_ids = (
                    encoded.input_ids
                    if hasattr(encoded, "input_ids")
                    else encoded["input_ids"]
                )
                lengths = [len(ids) for ids in input_ids]
            else:
                lengths = [0] * len(pending_offsets)
            context_accepted = [
                (not filter_long_context) or (0 < length <= max_context_tokens)
                for length in lengths
            ]
            accepted = context_accepted
            if tokenizer is not None:
                flat_questions = [text for row in pending_questions for text in row]
                flat_answers = [text for row in pending_answers for text in row]
                question_encoded = tokenizer(
                    flat_questions,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_question_tokens + 1,
                    padding=False,
                )
                question_ids = (
                    question_encoded.input_ids
                    if hasattr(question_encoded, "input_ids")
                    else question_encoded["input_ids"]
                )
                answer_encoded = tokenizer(
                    flat_answers,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_answer_tokens + 1,
                    padding=False,
                )
                answer_ids = (
                    answer_encoded.input_ids
                    if hasattr(answer_encoded, "input_ids")
                    else answer_encoded["input_ids"]
                )
                if len(question_ids) != len(flat_questions) or len(answer_ids) != len(
                    flat_answers
                ):
                    raise ValueError(
                        "tokenizer returned a different number of rows for QA filtering"
                    )
                accepted = []
                cursor = 0
                for keep, questions, answers in zip(
                    context_accepted, pending_questions, pending_answers
                ):
                    row_keep = keep
                    question_total = 0
                    answer_total = 0
                    for index in range(len(questions)):
                        row_keep = row_keep and (
                            0
                            < len(question_ids[cursor + index])
                            <= self.max_question_tokens
                            and 0
                            < len(answer_ids[cursor + index])
                            <= self.max_answer_tokens
                        )
                        question_total += len(question_ids[cursor + index])
                        answer_total += len(answer_ids[cursor + index])
                    # The conversation budget applies uniformly to every row. For a
                    # single-QA row this is naturally above the individual Q+A limit.
                    row_keep = row_keep and question_total + answer_total <= self.max_conversation_tokens
                    accepted.append(row_keep)
                    cursor += len(questions)
            accepted_offsets = [
                offset for offset, keep in zip(pending_offsets, accepted) if keep
            ]
            reached_limit = False
            for offset in accepted_offsets:
                offsets.append(offset)
                if max_samples is not None and len(offsets) >= max_samples:
                    reached_limit = True
                    break
            pending_offsets.clear()
            pending_contexts.clear()
            pending_questions.clear()
            pending_answers.clear()
            return reached_limit

        total_bytes = self.path.stat().st_size
        progress = tqdm(
            total=total_bytes,
            desc="Indexing conversation data",
            unit="B",
            unit_scale=True,
            disable=not _progress_enabled(),
        )

        try:
            with self.path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    progress.update(handle.tell() - offset)
                    if not line:
                        break
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    context = str(row.get("context", "")).strip()
                    if not context:
                        continue
                    messages = _messages_from_row(row)
                    if filter_no_conversation and not any(
                        message.role == "assistant" for message in messages
                    ):
                        continue
                    if tokenizer is not None:
                        turns = ContextRecord("", messages).turns()
                        if not turns:
                            continue
                        if getattr(tokenizer, "apply_chat_template", None):
                            question_texts = [
                                render_conversation(tokenizer, turn.messages)
                                for turn in turns
                            ]
                        else:
                            question_texts = [turn.question for turn in turns]
                        pending_offsets.append(offset)
                        pending_contexts.append(context)
                        pending_questions.append(question_texts)
                        pending_answers.append([turn.answer for turn in turns])
                        if len(pending_offsets) >= batch_size:
                            if flush_tokenizer_batch():
                                break
                    else:
                        offsets.append(offset)
                    if max_samples is not None and len(offsets) >= max_samples:
                        break
                if pending_offsets:
                    flush_tokenizer_batch()
        finally:
            progress.close()
        if max_samples is not None:
            offsets = offsets[:max_samples]
        return offsets

    def _load_or_build_offsets(
        self,
        tokenizer,
        max_context_tokens,
        max_input_tokens,
        max_samples,
        filter_long_context,
        filter_no_conversation,
        cache_dir: Path | None,
    ) -> list[int]:
        metadata = self._cache_metadata(tokenizer)
        if cache_dir is None:
            return self._build_offsets(
                tokenizer,
                max_context_tokens,
                max_input_tokens,
                max_samples,
                filter_long_context,
                filter_no_conversation,
            )
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._offset_cache_path(cache_dir, metadata)
        with cache_utils.file_lock(path.with_suffix(".lock")):
            offsets = self._load_offsets(path, metadata) if path.exists() else None
            if offsets is None:
                offsets = self._build_offsets(
                    tokenizer,
                    max_context_tokens,
                    max_input_tokens,
                    max_samples,
                    filter_long_context,
                    filter_no_conversation,
                )
                self._save_offsets(path, metadata, offsets)
                if _progress_enabled():
                    print(
                        f"Saved conversation index cache: {path} "
                        f"(contexts={len(offsets)})",
                        flush=True,
                    )
            elif _progress_enabled():
                print(
                    f"Reusing conversation index cache: {path} "
                    f"(contexts={len(offsets)})",
                    flush=True,
                )
            return offsets

    def __len__(self):
        return len(self._offsets)

    def __getitem__(self, index):
        if self._handle is None:
            self._handle = self.path.open("rb")
        self._handle.seek(self._offsets[index])
        row = json.loads(self._handle.readline())
        messages = _messages_from_row(row)
        return ContextRecord(
            str(row["context"]).strip(),
            messages,
            self.path.stem,
            str(row.get("context_id", row.get("id", ""))),
        )


class TrainingTaskDataset(Dataset):
    """Context-level training view; task expansion is performed by the collator."""

    def __init__(self, dataset, mode="mix", recon_ratio=0.5):
        if mode not in {"mix", "qa", "recon"} or not 0 <= recon_ratio <= 1:
            raise ValueError("invalid training task mixture")
        self.source, self.mode, self.recon_ratio = dataset, mode, recon_ratio
        self.cache_metadata = {
            "source": dataset.cache_metadata,
            "mode": mode,
            "recon_ratio": recon_ratio,
        }

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        if self.mode == "mix":
            context_index, task = divmod(index, 2)
            row = self.source[context_index]
            return ContextRecord(
                row.context,
                row.conversations,
                row.dataset,
                row.context_id,
                row.context_token_length,
                "recon" if task else "qa",
            )
        return self.source[index]

    def sample_indices(self, seed):
        """Retained for callers inspecting historical candidate ratios; unused in training."""
        if self.mode != "mix":
            return list(range(len(self.source)))
        rng = random.Random(seed)
        n = len(self.source)
        recon = round(n * self.recon_ratio)
        qa = list(range(0, 2 * n, 2))
        rc = list(range(1, 2 * n, 2))
        rng.shuffle(qa)
        rng.shuffle(rc)
        return qa[: n - recon] + rc[:recon]


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
    ):
        self.dataset, self.batch_size, self.seed, self.epoch = (
            dataset,
            batch_size,
            seed,
            0,
        )
        metadata = self._cache_metadata(tokenizer, chat_template_enable_thinking)
        self.lengths = self._load_or_build_lengths(
            tokenizer,
            chat_template_enable_thinking,
            Path(cache_dir) if cache_dir else None,
            metadata,
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

    def _build_lengths(self, tokenizer, enable_thinking):
        lengths = []
        batch_size = max(
            1, int(getattr(self.dataset, "tokenizer_filter_batch_size", 1024))
        )
        pending_contexts = []
        progress = tqdm(
            total=len(self.dataset),
            desc="Preparing sortish lengths",
            unit="context",
            disable=not _progress_enabled(),
        )

        def flush() -> None:
            if not pending_contexts:
                return
            encoded = tokenizer(
                pending_contexts,
                add_special_tokens=False,
                truncation=False,
                padding=False,
            )
            input_ids = (
                encoded.input_ids
                if hasattr(encoded, "input_ids")
                else encoded["input_ids"]
            )
            if len(input_ids) != len(pending_contexts):
                raise ValueError(
                    "tokenizer returned a different number of rows for sortish lengths"
                )
            lengths.extend(len(ids) for ids in input_ids)
            progress.update(len(pending_contexts))
            pending_contexts.clear()

        try:
            for record in self.dataset:
                pending_contexts.append(record.context)
                if len(pending_contexts) >= batch_size:
                    flush()
            flush()
        finally:
            progress.close()
        return lengths

    def _load_or_build_lengths(self, tokenizer, enable_thinking, cache_dir, metadata):
        if cache_dir is None:
            return self._build_lengths(tokenizer, enable_thinking)
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
            lengths = self._build_lengths(tokenizer, enable_thinking)
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
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return sum(len(batch) for batch in self.buckets)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._rebuild()


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


def collate_context_records(
    rows,
    tokenizer,
    max_context_tokens=2048,
    max_question_tokens=96,
    max_answer_tokens=64,
    append_eos=True,
    chat_template_enable_thinking=False,
    qa_per_context=4,
    sample_qa=True,
    question_padding_side="left",
    training=False,
    train_tasks="mix",
):
    """Build memory-prefix QA batches from structured conversations.

    Every selected turn is rendered as ``history + current user + generation prompt``.
    The current assistant content is kept as the target, so PWC history is visible while
    the answer being supervised is never leaked in the input.
    """
    if not getattr(tokenizer, "apply_chat_template", None):
        raise ValueError("conversation training requires tokenizer.apply_chat_template")
    if train_tasks not in {"mix", "qa", "recon"}:
        raise ValueError("train_tasks must be qa, recon or mix")
    contexts = [row.context for row in rows]
    c = _encode(tokenizer, contexts, max_context_tokens)
    selected: list[ConversationTurn] = []
    mapping: list[int] = []
    task_ids: list[int] = []
    for row_index, record in enumerate(rows):
        turns = list(record.turns())
        if training and train_tasks in {"qa", "mix"} and not turns:
            raise ValueError(
                "conversation training requires at least one assistant turn"
            )
        if training and train_tasks == "mix":
            chosen = [
                random.choice(turns),
                ConversationTurn(
                    (),
                    record.context,
                    dataset=record.dataset,
                    context_id=record.context_id,
                ),
            ]
            chosen_tasks = [1, 0]  # 1=QA, 0=Recon
        elif training and train_tasks == "recon":
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
            chosen = [random.choice(turns)]
            chosen_tasks = [1]
        else:
            chosen = (
                random.sample(turns, min(qa_per_context, len(turns)))
                if sample_qa
                else turns
            )
            chosen_tasks = [1] * len(chosen)
        selected.extend(chosen)
        mapping.extend([row_index] * len(chosen))
        task_ids.extend(chosen_tasks)
    if not selected:
        raise ValueError("conversation batch contains no assistant turns")

    rendered = []
    for turn, task in zip(selected, task_ids):
        if task == 0:
            rendered.append("")
            continue
        messages = turn.messages
        payload = [{"role": item.role, "content": item.content} for item in messages]
        try:
            text = tokenizer.apply_chat_template(
                payload,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=chat_template_enable_thinking,
            )
        except (TypeError, ValueError):
            text = tokenizer.apply_chat_template(
                payload, tokenize=False, add_generation_prompt=True
            )
        rendered.append(text)
    # The template already contains role markers and the assistant generation marker.
    qa_rendered = [text for text, task in zip(rendered, task_ids) if task == 1]
    if qa_rendered:
        previous_truncation = tokenizer.truncation_side
        tokenizer.truncation_side = "left"
        try:
            qa_encoded = _encode(
                tokenizer,
                qa_rendered,
                max_question_tokens,
                padding_side=question_padding_side,
            )
        finally:
            tokenizer.truncation_side = previous_truncation
        q = torch.zeros((len(selected), qa_encoded.input_ids.size(1)), dtype=qa_encoded.input_ids.dtype)
        q_mask = torch.zeros((len(selected), qa_encoded.attention_mask.size(1)), dtype=torch.bool)
        qa_rows = torch.tensor([task == 1 for task in task_ids])
        q[qa_rows] = qa_encoded.input_ids
        q_mask[qa_rows] = qa_encoded.attention_mask.bool()
    else:
        q = torch.empty((len(selected), 0), dtype=c.input_ids.dtype)
        q_mask = torch.empty((len(selected), 0), dtype=torch.bool)

    target_rows = [
        tokenizer(
            turn.answer,
            truncation=True,
            max_length=max_context_tokens if task == 0 else max_answer_tokens,
            add_special_tokens=False,
        ).input_ids
        for turn, task in zip(selected, task_ids)
    ]
    previous = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        a = tokenizer.pad({"input_ids": target_rows}, padding=True, return_tensors="pt")
    finally:
        tokenizer.padding_side = previous
    if (
        append_eos
        and any(task == 1 for task in task_ids)
        and getattr(tokenizer, "eos_token_id", None) is not None
    ):
        eos = int(tokenizer.eos_token_id)
        rows_n, width = a.input_ids.size(0), a.input_ids.size(1) + 1
        ids = torch.full(
            (rows_n, width), int(tokenizer.pad_token_id), dtype=a.input_ids.dtype
        )
        mask = torch.zeros((rows_n, width), dtype=torch.long)
        for index in range(rows_n):
            valid = int(a.attention_mask[index].sum())
            ids[index, :valid] = a.input_ids[index, :valid]
            mask[index, :valid] = 1
            if task_ids[index] == 1:
                ids[index, valid] = eos
                mask[index, valid] = 1
        a["input_ids"] = ids
        a["attention_mask"] = mask
    labels = torch.full_like(a.input_ids, -100)
    labels[a.attention_mask.bool()] = a.input_ids[a.attention_mask.bool()]
    batch = {
        "context_ids": c.input_ids,
        "context_mask": c.attention_mask.bool(),
        "input_ids": q,
        "input_mask": q_mask,
        "target_ids": a.input_ids,
        "target_mask": a.attention_mask.bool(),
        "labels": labels,
        "qa_context_indices": torch.tensor(mapping, dtype=torch.long),
        "records": selected,
    }
    if training:
        batch["task_ids"] = torch.tensor(task_ids, dtype=torch.long)
    return batch


# Public name for the collate. It used to be a pass-through wrapper that re-declared all
# twelve parameters and defaults -- which is exactly how a default silently drifts away
# from the real signature (the wrapper still said question_padding_side="right" /
# legacy padding defaults long after the real defaults changed).
collate_fn = collate_context_records
