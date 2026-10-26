"""Config-driven assembly of datasets and collate callables."""
from __future__ import annotations

from pathlib import Path

from .cache import cache_is_fresh
from .collate import collate_fn
from .dataset import ContextDataset
from .preprocessing import progress_enabled


def data_cache_dir(cfg) -> Path:
    """Return the loader-side preprocessing cache directory."""
    if cfg.data.preprocess_cache_dir:
        return Path(cfg.data.preprocess_cache_dir)
    return Path(cfg.data.root) / "1qa" / "cache" / "loader"

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

def uses_filtered_cache(cfg, split: str, tokenizer) -> bool:
    """True when this split's file is the *fresh* output of ``build_cache.py``.

    Then the loader may read it directly: the cache fingerprint hashes the source meta file,
    the tokenizer and the filter options, so a match means "the rules that produced this file
    are the rules this config asks for". A stale, hand-made or overridden file returns False and
    the loader filters it as before.
    """
    if split not in {"train", "validation"}:
        return False
    if Path(cfg.data.split_file(split)) != Path(cfg.data.cache_file(split)):
        return False
    return cache_is_fresh(cfg.data, split, tokenizer)

def make_context_dataset(
    cfg,
    split: str,
    tokenizer,
    limit=None,
    allow_empty: bool = False,
    num_workers: int = 0,
):
    """Build one split. ``limit`` is the resolved sample cap (see :func:`resolve_split_limit`).

    The path comes from ``DataConfig.split_file``: an explicit
    ``data.files.<split>_conversation_file`` when set, otherwise
    ``<data.files.root>/1qa/1qa-<split>.jsonl`` (the layout ``scripts/data/`` writes).
    """
    path = cfg.data.split_file(split)
    pre_filtered = uses_filtered_cache(cfg, split, tokenizer)
    if pre_filtered and progress_enabled():
        print(
            f"[data] {split}: {Path(path).name} is this config's filtered cache "
            "(loader-side re-filter skipped)",
            flush=True,
        )
    cache_dir = data_cache_dir(cfg) if cfg.data.cache_filtered_dataset else None
    return ContextDataset(
        path,
        tokenizer=tokenizer,
        max_context_tokens=cfg.data.max_context_tokens,
        max_input_tokens=cfg.data.max_input_tokens,
        max_samples=limit,
        filter_long_context=cfg.data.filter_long_context,
        filter_no_conversation=cfg.data.filter_no_qa,
        cache_dir=cache_dir,
        tokenizer_filter_batch_size=cfg.data.tokenizer_filter_batch_size,
        max_question_tokens=cfg.data.max_question_tokens,
        max_answer_tokens=cfg.data.max_answer_tokens,
        filter_qa_pair_answers=cfg.data.filter_qa_pair_answers,
        allow_empty=allow_empty,
        num_workers=num_workers,
        pre_filtered=pre_filtered,
    )

def make_collate(cfg, tokenizer, sample_qa: bool, training=False):
    """Keyword-only collate: positional drift would change the target format silently.

    Returns a callable that also carries the sampling epoch, so the QA-turn subset of a
    context can follow the sampler's epoch (``Collate.set_epoch``) instead of the process
    global RNG, which depended on the DataLoader worker layout.
    """
    return Collate(cfg, tokenizer, sample_qa=sample_qa, training=training)

class Collate:
    """One collate callable per loader, with an epoch the caller can advance."""

    def __init__(self, cfg, tokenizer, sample_qa: bool, training: bool = False):
        self.cfg, self.tokenizer = cfg, tokenizer
        self.sample_qa, self.training = bool(sample_qa), bool(training)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> Collate:
        self.epoch = int(epoch)
        return self

    def __call__(self, rows):
        cfg, tokenizer = self.cfg, self.tokenizer
        train_stage = cfg.data.train_stage
        if self.training and train_stage == "two_stage":
            train_stage = (
                "only_recon" if self.epoch < cfg.training.epochs else "only_qa"
            )
        return collate_fn(
            rows,
            tokenizer,
            max_context_tokens=cfg.data.max_context_tokens,
            max_question_tokens=cfg.data.max_question_tokens,
            max_answer_tokens=cfg.data.max_answer_tokens,
            append_eos=cfg.data.append_eos,
            chat_template_enable_thinking=cfg.data.chat_template_enable_thinking,
            max_qa_sample_per_context=cfg.data.max_qa_sample_per_context,
            sample_qa=self.sample_qa,
            question_padding_side=cfg.data.question_padding_side,
            training=self.training,
            train_stage=train_stage if self.training else "only_qa",
            epoch=self.epoch,
        )

__all__ = [
    "Collate",
    "ContextDataset",
    "data_cache_dir",
    "make_collate",
    "make_context_dataset",
    "resolve_split_limit",
    "uses_filtered_cache",
]
