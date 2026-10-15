"""Seeded per-dataset QA samples shared by QA and Recon, invariant to GPU count."""
from __future__ import annotations

import json
from pathlib import Path

from src.data import ConversationDataset
from src.pipeline import (
    evaluation_row_index,
    evaluation_window_loader,
    select_evaluation_samples,
    window_slice,
)


class _Record:
    """A dataset element that only has to answer ``turns()``."""

    def __init__(self, turns: int, context: str = "c"):
        self._turns = turns
        self.context = context

    def turns(self):
        return tuple(range(self._turns))


class _Dataset:
    def __init__(self, turn_counts):
        self.turn_counts = list(turn_counts)

    def __len__(self):
        return len(self.turn_counts)

    def __getitem__(self, index):
        return _Record(self.turn_counts[index], f"c{index}")


class _Encoded:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class _Tokenizer:
    """Just enough tokenizer for the filter in ``ConversationDataset``.

    Flat ids for a single string, a list of them for a batch -- the real contract; a
    batch-shaped answer for a single string made the length rule read ``len([[tokens]]) == 1``.
    """

    name_or_path = "window-test"
    vocab_size = 32

    def __call__(self, texts, **kwargs):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        limit = kwargs.get("max_length")
        encoded = [text.split()[:limit] if limit else text.split() for text in items]
        return _Encoded(encoded[0] if single else encoded)


def test_the_row_index_counts_the_turns_of_each_context():
    assert evaluation_row_index(_Dataset([2, 3, 1])) == [0, 2, 5, 6]
    assert evaluation_row_index(_Dataset([])) == [0]


def test_a_row_window_maps_to_the_contexts_that_cover_it():
    index = [0, 2, 5, 6]
    assert window_slice(index, (0, 2)) == (0, 1)
    assert window_slice(index, (2, 5)) == (1, 2)
    # A window that starts inside a context still needs that whole context ...
    assert window_slice(index, (3, 6)) == (1, 3)
    # ... and one that starts at the very end returns a single-element range, not an empty one.
    assert window_slice(index, (0, None)) == (0, 3)
    assert window_slice(index, (5, 5)) == (2, 2)


def test_the_windowed_loader_fetches_only_the_window(tmp_path: Path):
    source = tmp_path / "conversation.jsonl"
    counts = [1, 2, 3, 1]
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "context": f"context number {index}",
                    "conversations": [
                        item
                        for turn in range(count)
                        for item in (
                            {"role": "user", "content": f"question {index} {turn}"},
                            {"role": "assistant", "content": f"answer {index} {turn}"},
                        )
                    ],
                }
            )
            for index, count in enumerate(counts)
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = ConversationDataset(
        source, tokenizer=_Tokenizer(), max_context_tokens=64, allow_empty=True,
    )
    assert evaluation_row_index(dataset) == [0, 1, 3, 6, 7]

    # Contexts remain whole while all ranks partition the same seeded QA sample.
    loader, base = evaluation_window_loader(
        dataset, lambda rows: rows, budget=4, unit="rows", seed=7,
        rank=0, world=2, batch_size=8,
    )
    selection = loader.evaluation_selection
    actual = []
    for rank in range(2):
        rows_loader, rows_base = evaluation_window_loader(
            dataset, lambda rows: rows, budget=4, unit="rows", seed=7,
            rank=rank, world=2, batch_size=8,
        )
        contexts_loader, contexts_base = evaluation_window_loader(
            dataset, lambda rows: rows, budget=4, unit="selected_contexts",
            selection=selection, rank=rank, world=2, batch_size=8,
        )
        assert rows_base == contexts_base == 0
        row_records = [record for batch in rows_loader for record in batch]
        recon_records = [record for batch in contexts_loader for record in batch]
        row_contexts = {
            int(record.context.rsplit(" ", 1)[-1]): tuple(record.turns())
            for record in row_records
        }
        recon_contexts = {
            int(record.context.rsplit(" ", 1)[-1]): tuple(record.turns())
            for record in recon_records
        }
        assert row_contexts == recon_contexts
        actual.extend(
            (context, dataset[context].turns().index(turn))
            for context, turns in row_contexts.items()
            for turn in turns
        )
    expected = [
        (context, turn)
        for context, turns in selection.contexts
        for turn in turns
    ]
    assert sorted(actual) == sorted(expected)


def test_an_unknown_budget_unit_is_rejected():
    try:
        evaluation_window_loader(
            _Dataset([1]), lambda rows: rows, budget=1, unit="tokens", rank=0, world=1,
            batch_size=1,
        )
    except ValueError as error:
        assert "tokens" in str(error)
    else:
        raise AssertionError("an unknown unit must not silently mean something else")


def test_per_dataset_random_samples_are_seeded_and_independent_of_world_size():
    # Interleaved datasets, contexts crossing the cap, and one small dataset.
    records = []
    for name, count in [("squad", 2), ("race", 1), ("squad", 4),
                        ("hotpotqa", 1), ("race", 5), ("squad", 1)]:
        record = _Record(count, f"{name}:{len(records)}")
        record.dataset = name
        records.append(record)

    selection = select_evaluation_samples(records, 3, 2, seed=42)
    assert selection == select_evaluation_samples(records, 3, 2, seed=42)
    assert selection != select_evaluation_samples(records, 3, 2, seed=43)
    assert selection.qa_rows <= len(selection.contexts) * 2
    context_counts = dict(selection.context_counts)
    qa_counts = dict(selection.qa_counts)
    assert all(count <= 3 for count in context_counts.values())
    assert all(qa_counts[name] <= 2 * count for name, count in context_counts.items())

    def selected(world):
        result = []
        for rank in range(world):
            loader, _ = evaluation_window_loader(
                records, list, budget=3, unit="rows", seed=42,
                max_qa_sample_per_context=2,
                rank=rank, world=world, batch_size=2,
            )
            assert loader.evaluation_total == selection.qa_rows
            result.extend(
                (record.context, turn)
                for batch in loader for record in batch for turn in record.turns()
            )
        return result

    expected = [
        (records[context].context, turn)
        for context, turns in selection.contexts for turn in turns
    ]
    for world in (1, 2, 4, 8):
        assert sorted(selected(world)) == sorted(expected)


def test_rank_sharding_keeps_conversation_turn_history():
    from src.data import ContextRecord, ConversationMessage

    record = ContextRecord(
        "context",
        tuple(ConversationMessage(role, text) for role, text in [
            ("user", "first question"), ("assistant", "first answer"),
            ("user", "second question"), ("assistant", "second answer"),
        ]),
        dataset="squad",
    )
    loader, base = evaluation_window_loader(
        [record], list, budget=2, unit="rows", seed=1,
        rank=0, world=2, batch_size=1,
    )
    assert base == 0
    turns = [
        turn for batch in loader for row in batch for turn in row.turns()
    ]
    assert turns == list(record.turns())


def test_empty_qa_selection_has_no_batches():
    loader, base = evaluation_window_loader(
        [], list, budget=3, unit="rows", seed=0, rank=0, world=4, batch_size=1,
    )
    assert loader.evaluation_total == base == 0
    assert list(loader) == []


def test_evaluator_scores_every_dataset_without_a_combined_cap():
    from types import SimpleNamespace

    import torch
    from torch.utils.data import DataLoader

    from src.data import ContextRecord, ConversationMessage
    from src.evaluator import Evaluator
    from utils.config import TrainConfig

    records = [
        ContextRecord("context", tuple(
            ConversationMessage(role, text)
            for _ in range(count)
            for role, text in [("user", "question"), ("assistant", "answer")]
        ), dataset=name)
        for name, count in [("squad", 5), ("hotpotqa", 1), ("squad", 2)]
    ]

    def collate(rows):
        turns = [turn for row in rows for turn in row.turns()]
        return {
            "context_ids": torch.ones(len(rows), 1, dtype=torch.long),
            "context_mask": torch.ones(len(rows), 1, dtype=torch.bool),
            "qa_input_ids": torch.ones(len(turns), 1, dtype=torch.long),
            "qa_input_mask": torch.ones(len(turns), 1, dtype=torch.bool),
            "qa_context_indices": torch.tensor([
                i for i, row in enumerate(rows) for _ in row.turns()
            ]),
            "records": turns,
        }

    class Model(torch.nn.Module):
        qwen = SimpleNamespace(get_input_embeddings=lambda: torch.nn.Embedding(4, 2))

        def encode_context_prefix(self, *args):
            return None

        def generate_answers_with_prefix(self, prefix, indices, *args, **kwargs):
            return torch.ones(len(indices), 1, dtype=torch.long)

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, *args, **kwargs):
            return SimpleNamespace(input_ids=[1])

        def decode(self, *args, **kwargs):
            return "answer"

    model = Model()
    loader = DataLoader(records, batch_size=1, collate_fn=collate)
    evaluator = Evaluator(Tokenizer(), TrainConfig())
    # Exercise both direct evaluator calls and the pre-windowed training/test path.
    for pre_windowed in (False, True):
        options = {}
        if pre_windowed:
            selected, base = evaluation_window_loader(
                records, collate, budget=3, unit="rows", seed=0,
                max_qa_sample_per_context=2,
                rank=0, world=1, batch_size=1,
            )
            options = dict(row_loader=selected, row_base=base, pre_windowed=True)
        metrics = evaluator.autoregressive(
            model, loader, torch.device("cpu"),
            max_contexts_per_dataset=3, max_qa_sample_per_context=2, **options,
        )
        assert metrics["dataset/squad/count"] == 4
        assert metrics["dataset/hotpotqa/count"] == 1
        assert metrics["em"] == 1
        assert model.training


def test_compare_reader_tasks_reuses_the_sampled_contexts_for_recon():
    import torch
    from torch.utils.data import DataLoader

    from src.data import ContextRecord, ConversationMessage
    from src.evaluator import Evaluator
    from utils.config import TrainConfig

    records = [
        ContextRecord(
            f"context-{name}-{i}",
            (ConversationMessage("user", "q"), ConversationMessage("assistant", "a")),
            dataset=name,
        )
        for name, count in (("squad", 4), ("race", 3))
        for i in range(count)
    ]
    cfg = TrainConfig()
    cfg.training.seed = 23
    evaluator = Evaluator(None, cfg)
    recon_seen = {}

    def autoregressive(*args, **kwargs):
        return {"f1": 1.0}

    def recon_teacher_forced(model, loader, device, row_loader=None, **kwargs):
        recon_seen["contexts"] = [
            record.context for batch in row_loader for record in batch
        ]
        recon_seen["qa_rows"] = sum(
            len(record.turns()) for batch in row_loader for record in batch
        )
        return {"contexts": float(len(recon_seen["contexts"]))}

    evaluator.autoregressive = autoregressive
    evaluator.recon_teacher_forced = recon_teacher_forced
    evaluator.compare_reader_tasks(
        object(),
        DataLoader(records, batch_size=2, collate_fn=list),
        torch.device("cpu"),
        max_contexts_per_dataset=2, max_qa_sample_per_context=1,
    )
    expected = select_evaluation_samples(records, 2, 1, seed=cfg.training.seed)
    assert recon_seen["contexts"] == [
        records[context].context for context, _ in expected.contexts
    ]
    assert recon_seen["qa_rows"] == expected.qa_rows
