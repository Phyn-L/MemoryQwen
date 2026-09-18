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

from .data import AggregatedContextDataset, collate_fn

MODEL_LABEL_PATTERN = re.compile(r"Qwen(?:3)?[-_]?([0-9]+(?:\.[0-9]+)?)[Bb]", re.IGNORECASE)


def model_run_label(model_path: str) -> str:
    """``.../models--Qwen--Qwen3-8B/snapshots/...`` -> ``Qwen8B``."""
    match = MODEL_LABEL_PATTERN.search(model_path)
    return f"Qwen{match.group(1)}B" if match else "Qwen"


def model_cache_dir(model_path: str) -> Path:
    """Per-model directory for the reusable dataset / sortish-length caches."""
    return Path("outputs") / model_run_label(model_path)


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


def make_context_dataset(cfg, split: str, tokenizer, limit=None, allow_empty: bool = False, source_version: str | None = None):
    """Build one split. ``limit`` is the resolved sample cap (see :func:`resolve_split_limit`)."""
    names = getattr(cfg.data, f"{split}_datasets") or cfg.data.dataset
    # Keep HotpotQA held out of the implicit training mixture for now.
    # Explicit selections and validation/test datasets remain available.
    if split == "train" and names in (None, "all"):
        names = [p.name for p in sorted(Path(cfg.data.root).iterdir())
                 if p.is_dir() and p.name.lower() != "hotpotqa"]
    split_name = getattr(cfg.data, f"{split}_split")
    cache_dir = model_cache_dir(cfg.model.name_or_path) if cfg.data.cache_dataset else None
    return AggregatedContextDataset(
        cfg.data.root, names, split_name, tokenizer,
        cfg.data.max_context_tokens, limit,
        cfg.data.filter_long_context, cfg.data.filter_no_qa,
        allow_empty=allow_empty, cache_dir=cache_dir, source_version=source_version,
    )


def make_collate(cfg, tokenizer, sample_qa: bool):
    """Keyword-only collate: positional drift would change the target format silently."""
    def collate(rows):
        return collate_fn(
            rows, tokenizer,
            max_context_tokens=cfg.data.max_context_tokens,
            max_question_tokens=cfg.data.max_question_tokens,
            max_answer_tokens=cfg.data.max_answer_tokens,
            append_eos=cfg.data.append_eos,
            use_chat_template=cfg.data.use_chat_template,
            chat_template_enable_thinking=cfg.data.chat_template_enable_thinking,
            qa_per_context=cfg.data.qa_per_context,
            sample_qa=sample_qa,
            question_padding_side=cfg.data.question_padding_side,
            eos_mode=cfg.data.eos_mode,
        )
    return collate
