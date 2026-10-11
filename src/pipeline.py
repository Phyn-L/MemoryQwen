"""Run-path plumbing shared by ``scripts/train.py`` and ``scripts/test.py``.

Both entry points used to carry their own copy of the model-label regex, the dataset
construction and the eleven-kwarg collate call. A change to the target format landing in
only one of them would silently make validation disagree with training, so there is one
copy here.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

from .data import ConversationDataset, collate_fn

MODEL_LABEL_PATTERN = re.compile(r"Qwen(?:3)?[-_]?([0-9]+(?:\.[0-9]+)?)[Bb]", re.IGNORECASE)


def model_run_label(model_path: str) -> str:
    """``.../models--Qwen--Qwen3-8B/snapshots/...`` -> ``Qwen8B``."""
    match = MODEL_LABEL_PATTERN.search(model_path)
    return f"Qwen{match.group(1)}B" if match else "Qwen"


def data_cache_dir() -> Path:
    """Shared cache directory for data preprocessing across compatible models."""
    return Path("outputs") / "conversation_cache"


def new_run_name(model_path: str) -> str:
    """``Qwen8B_20260916_213847`` — used for the W&B run and the checkpoint directory."""
    return f"{model_run_label(model_path)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def resolve_split_limit(cfg, split: str) -> int | None:
    """Config-provided sample cap for a split.

    ``{split}_max_samples`` first, then ``data.max_samples`` for the train split. Kept as
    an explicit resolver rather than a default inside :func:`make_context_dataset` so a
    caller with its own cap (``scripts/test.py --max-samples``) cannot accidentally pick
    up a config limit it did not ask for.
    """
    limit = getattr(cfg.data, f"{split}_max_samples", None)
    if limit is None and split == "train":
        limit = cfg.data.max_samples
    return limit


def make_context_dataset(cfg, split: str, tokenizer, limit=None, allow_empty: bool = False):
    """Build one split. ``limit`` is the resolved sample cap (see :func:`resolve_split_limit`)."""
    path = (
        getattr(cfg.data, f"{split}_conversation_file", None)
        or getattr(cfg.data, "conversation_file", None)
    )
    if not path:
        raise ValueError(
            f"data.{split}_conversation_file or data.conversation_file is required"
        )
    cache_dir = data_cache_dir() if cfg.data.cache_offsets else None
    return ConversationDataset(
        path,
        tokenizer=tokenizer,
        max_context_tokens=cfg.data.max_context_tokens,
        max_input_context_length=cfg.data.max_input_context_length,
        max_samples=limit,
        filter_long_context=cfg.data.filter_long_context,
        filter_no_conversation=cfg.data.filter_no_qa,
        cache_dir=cache_dir,
        tokenizer_filter_batch_size=cfg.data.tokenizer_filter_batch_size,
        allow_empty=allow_empty,
    )


def make_collate(cfg, tokenizer, sample_qa: bool, training=False):
    """Keyword-only collate: positional drift would change the target format silently."""
    def collate(rows):
        return collate_fn(
            rows, tokenizer,
            max_context_tokens=cfg.data.max_context_tokens,
            max_conversation_tokens=cfg.data.max_conversation_tokens,
            max_answer_tokens=cfg.data.max_answer_tokens,
            append_eos=cfg.data.append_eos,
            chat_template_enable_thinking=cfg.data.chat_template_enable_thinking,
            qa_per_context=cfg.data.qa_per_context,
            sample_qa=sample_qa,
            question_padding_side=cfg.data.question_padding_side,
            training=training,
            train_tasks=cfg.data.train_tasks if training else "qa",
            recon_prompt=cfg.data.recon_prompt,
        )
    return collate
