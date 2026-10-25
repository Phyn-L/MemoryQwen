"""Deterministic evaluation sampling and rank-local loader construction."""

from __future__ import annotations

import bisect
import random
from collections.abc import Sequence
from dataclasses import dataclass

from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EvaluationContext:
    """Keep selected turns with their original conversational history."""

    context: str
    selected_turns: tuple

    def turns(self):
        return self.selected_turns


class EvaluationRows:
    def __init__(self, dataset, selection):
        self.dataset, self.selection = dataset, selection

    def __len__(self):
        return len(self.selection)

    def __getitem__(self, index):
        context, turn_indices = self.selection[index]
        record = self.dataset[context]
        turns = record.turns()
        return EvaluationContext(record.context, tuple(turns[i] for i in turn_indices))


@dataclass(frozen=True)
class EvaluationSelection:
    """A deterministic QA sample and its unique source contexts."""

    contexts: tuple[tuple[int, tuple[int, ...]], ...]
    qa_rows: int
    qa_counts: tuple[tuple[str, int], ...]
    context_counts: tuple[tuple[str, int], ...]


def evaluation_row_index(dataset) -> list[int]:
    """Cumulative evaluation QA rows before each context."""
    offsets = [0]
    total = 0
    for record in dataset:
        total += len(record.turns())
        offsets.append(total)
    return offsets


def window_slice(
    index: Sequence[int], window: tuple[int, int | None]
) -> tuple[int, int]:
    """Dataset index range whose rows (or contexts) cover ``window``."""
    start, stop = window
    if stop is None:
        return 0, len(index) - 1
    low = max(0, bisect.bisect_right(index, int(start)) - 1)
    high = min(len(index) - 1, bisect.bisect_left(index, int(stop)))
    return min(low, high), max(low, high)


def select_evaluation_samples(
    dataset,
    max_contexts_per_dataset: int,
    max_qa_sample_per_context: int,
    seed: int,
) -> EvaluationSelection:
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
    return EvaluationSelection(
        contexts=tuple(by_context),
        qa_rows=sum(qa_counts.values()),
        qa_counts=tuple(sorted(qa_counts.items())),
        context_counts=tuple(sorted(context_counts.items())),
    )


def balanced_context_shards(dataset, selection, world: int, max_new_tokens: int):
    """Assign whole contexts to the least-loaded rank using a deterministic cost estimate."""
    shards = [[] for _ in range(max(1, int(world)))]
    loads = [0] * len(shards)
    entries = []
    for context, turn_indices in selection.contexts:
        text_length = max(1, len(str(dataset[context].context)))
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
    """Load the sampled QA rows or their unique contexts for one rank."""
    if unit == "rows":
        selection = selection or select_evaluation_samples(
            dataset,
            budget,
            max_qa_sample_per_context,
            seed,
        )
        shards = balanced_context_shards(dataset, selection, world, max_new_tokens)
        grouped = shards[min(max(0, int(rank)), len(shards) - 1)]
        selected = EvaluationRows(
            dataset, tuple((context, tuple(indices)) for context, indices in grouped)
        )
        base = 0
        total = selection.qa_rows
        local_rows = sum(len(indices) for _, indices in grouped)
    elif unit == "selected_contexts":
        if selection is None:
            raise ValueError("selected_contexts requires the QA evaluation selection")
        shards = balanced_context_shards(dataset, selection, world, max_new_tokens)
        local_selection = shards[min(max(0, int(rank)), len(shards) - 1)]
        selected = EvaluationRows(dataset, local_selection)
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


__all__ = [
    "EvaluationContext",
    "EvaluationRows",
    "EvaluationSelection",
    "balanced_context_shards",
    "evaluation_row_index",
    "evaluation_window_loader",
    "select_evaluation_samples",
    "window_slice",
]
