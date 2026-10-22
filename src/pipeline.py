"""Run-path plumbing shared by ``scripts/train.py`` and ``scripts/test.py``.

Both entry points used to carry their own copy of the model-label regex, the dataset
construction and the eleven-kwarg collate call. A change to the target format landing in
only one of them would silently make validation disagree with training, so there is one
copy here.
"""

from __future__ import annotations

import bisect
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from torch.utils.data import DataLoader

from .cache_build import cache_is_fresh
from .data import ConversationDataset, collate_fn, progress_enabled

MODEL_LABEL_PATTERN = re.compile(
    r"Qwen(?:3)?[-_]?([0-9]+(?:\.[0-9]+)?)[Bb]", re.IGNORECASE
)


def model_run_label(model_path: str) -> str:
    """``.../models--Qwen--Qwen3-8B/snapshots/...`` -> ``Qwen8B``."""
    match = MODEL_LABEL_PATTERN.search(model_path)
    return f"Qwen{match.group(1)}B" if match else "Qwen"


def data_cache_dir(cfg) -> Path:
    """Where the loader's own preprocessing caches go.

    Default: ``<data.root>/1qa/cache/loader`` -- next to the data they were derived from, so
    every run and every user of that data tree reuses them and clearing ``outputs/`` does not
    throw them away. ``data.preprocess_cache_dir`` overrides it; point that at local scratch when
    the data root is a slow network filesystem (the sortish pass reads every context once, and a
    filtered-row cache is rewritten whenever the input file or the filter options change).
    """
    if cfg.data.preprocess_cache_dir:
        return Path(cfg.data.preprocess_cache_dir)
    return Path(cfg.data.root) / "1qa" / "cache" / "loader"


def new_run_name(model_path: str) -> str:
    """``Qwen8B_20260916_213847`` — default experiment name and checkpoint directory."""
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
    return ConversationDataset(
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


def evaluation_row_index(dataset) -> list[int]:
    """Cumulative evaluation QA rows before each context (the evaluation collate's order).

    The collate turns one context into one row per assistant turn, so the global QA row of a
    context is its turn count. Reading ``turns()`` costs no tokenization, which is what makes
    this index cheap enough to build once per run.
    """
    offsets = [0]
    total = 0
    for record in dataset:
        total += len(record.turns())
        offsets.append(total)
    return offsets


def window_slice(
    index: Sequence[int], window: tuple[int, int | None]
) -> tuple[int, int]:
    """Dataset index range whose rows (or contexts) cover ``window``.

    ``index[i]`` is the first row of dataset element ``i`` and ``index[-1]`` the total row
    count, so this is a pair of binary searches: the first element that still contributes to
    the window and the one past its end.
    """
    start, stop = window
    if stop is None:
        return 0, len(index) - 1
    low = max(0, bisect.bisect_right(index, int(start)) - 1)
    high = min(len(index) - 1, bisect.bisect_left(index, int(stop)))
    return min(low, high), max(low, high)


@dataclass(frozen=True)
class _EvaluationContext:
    """Keep selected turns with their original conversational history."""

    context: str
    selected_turns: tuple

    def turns(self):
        return self.selected_turns


class _EvaluationRows:
    def __init__(self, dataset, selection):
        self.dataset, self.selection = dataset, selection

    def __len__(self):
        return len(self.selection)

    def __getitem__(self, index):
        context, turn_indices = self.selection[index]
        record = self.dataset[context]
        turns = record.turns()
        return _EvaluationContext(record.context, tuple(turns[i] for i in turn_indices))


@dataclass(frozen=True)
class _EvaluationSelection:
    """A deterministic QA sample and its unique source contexts."""

    contexts: tuple[tuple[int, tuple[int, ...]], ...]
    qa_rows: int
    qa_counts: tuple[tuple[str, int], ...]
    context_counts: tuple[tuple[str, int], ...]


def select_evaluation_samples(
    dataset,
    max_contexts_per_dataset: int,
    max_qa_sample_per_context: int,
    seed: int,
) -> _EvaluationSelection:
    """Sample contexts per dataset, then cap selected QA turns within each context."""
    candidates = {}
    for context, record in enumerate(dataset):
        by_dataset = {}
        for index, turn in enumerate(record.turns()):
            name = str(getattr(turn, "dataset", "") or getattr(record, "dataset", ""))
            by_dataset.setdefault(name, []).append(index)
        for name, indices in by_dataset.items():
            candidates.setdefault(name, []).append((context, tuple(indices)))

    chosen_by_context = {}
    context_datasets = {}
    for name in sorted(candidates):
        contexts = candidates[name]
        rng = random.Random(f"{int(seed)}:{name}:contexts")
        sampled = rng.sample(
            contexts,
            min(int(max_contexts_per_dataset), len(contexts)),
        )
        for context, turn_indices in sampled:
            turn_rng = random.Random(f"{int(seed)}:{name}:{context}:turns")
            chosen_turns = turn_rng.sample(
                turn_indices,
                min(int(max_qa_sample_per_context), len(turn_indices)),
            )
            chosen_by_context.setdefault(context, []).extend(chosen_turns)
            context_datasets.setdefault(context, set()).add(name)

    by_context = [
        (context, tuple(sorted(set(turn_indices))))
        for context, turn_indices in sorted(chosen_by_context.items())
    ]
    qa_counts = {}
    context_counts = {}
    for context, turn_indices in by_context:
        record = dataset[context]
        turns = record.turns()
        record_name = str(getattr(record, "dataset", ""))
        for context_name in context_datasets[context]:
            context_counts[context_name] = context_counts.get(context_name, 0) + 1
        for turn_index in turn_indices:
            turn = turns[turn_index]
            name = str(getattr(turn, "dataset", "") or record_name)
            qa_counts[name] = qa_counts.get(name, 0) + 1
    return _EvaluationSelection(
        contexts=tuple(by_context),
        qa_rows=sum(qa_counts.values()),
        qa_counts=tuple(sorted(qa_counts.items())),
        context_counts=tuple(sorted(context_counts.items())),
    )


def _balanced_context_shards(dataset, selection, world: int, max_new_tokens: int):
    """Assign whole contexts to the least-loaded rank using a deterministic cost estimate."""
    shards = [[] for _ in range(max(1, int(world)))]
    loads = [0] * len(shards)
    entries = []
    for context, turn_indices in selection.contexts:
        text_length = max(1, len(str(dataset[context].context)))
        # Context encoding is quadratic in input length; answer decoding scales with
        # selected QA rows and the generation budget. Character length is a cheap,
        # tokenizer-independent proxy available on every dataset record.
        cost = text_length * text_length + (
            len(turn_indices) * max(1, int(max_new_tokens)) * 256
        )
        entries.append((cost, context, turn_indices))
    for cost, context, turn_indices in sorted(
        entries, key=lambda item: (-item[0], item[1])
    ):
        rank = min(range(len(shards)), key=lambda index: (loads[index], index))
        shards[rank].append((context, turn_indices))
        loads[rank] += cost
    for shard in shards:
        shard.sort(key=lambda item: item[0])
    return shards


def evaluation_window_loader(
    dataset,
    collate,
    *,
    budget,
    unit,
    rank,
    world,
    batch_size,
    seed=0,
    selection=None,
    max_qa_sample_per_context=8,
    max_new_tokens=64,
    num_workers=0,
    pin_memory=False,
    prefetch_factor=2,
    persistent_workers=False,
):
    """Load the sampled QA rows or their unique contexts for one rank.

    Selection is deterministic and global; rank only partitions that fixed sample.
    ``selected_contexts`` reuses the QA sample for Recon.
    """
    if unit == "rows":
        selection = selection or select_evaluation_samples(
            dataset,
            budget,
            max_qa_sample_per_context,
            seed,
        )
        shards = _balanced_context_shards(
            dataset,
            selection,
            world,
            max_new_tokens,
        )
        grouped = shards[min(max(0, int(rank)), len(shards) - 1)]
        selected = _EvaluationRows(
            dataset, tuple((context, tuple(indices)) for context, indices in grouped)
        )
        base = 0
        total = selection.qa_rows
        local_rows = sum(len(indices) for _, indices in grouped)
    elif unit == "selected_contexts":
        if selection is None:
            raise ValueError("selected_contexts requires the QA evaluation selection")
        shards = _balanced_context_shards(
            dataset,
            selection,
            world,
            max_new_tokens,
        )
        local_selection = shards[min(max(0, int(rank)), len(shards) - 1)]
        selected = _EvaluationRows(dataset, local_selection)
        base = 0
        total = len(selection.contexts)
        local_rows = len(local_selection)
    else:
        raise ValueError(f"unknown evaluation budget unit: {unit!r}")
    loader = DataLoader(
        selected,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers else None,
        persistent_workers=persistent_workers and num_workers > 0,
    )
    loader.evaluation_total = total
    loader.evaluation_selection = selection
    loader.evaluation_source_dataset = dataset
    loader.evaluation_collate = collate
    loader.evaluation_prepartitioned = True
    loader.evaluation_local_rows = local_rows
    return loader, base
