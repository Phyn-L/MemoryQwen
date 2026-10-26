import json
from pathlib import Path

from src.data.cache import tokenizer_fingerprint
from src.data.dataset import ContextDataset


class _Tokenizer:
    name_or_path = "cache-test-tokenizer"
    vocab_size = 32

    def __call__(self, text, **kwargs):
        if isinstance(text, list):
            class BatchEncoded:
                input_ids = [item.split() for item in text]

            return BatchEncoded()

        class Encoded:
            input_ids = text.split()

        return Encoded()

    def apply_chat_template(self, messages, **kwargs):
        return " ".join(item["content"] for item in messages)


class _Backend:
    def to_str(self):
        return '{"model": "qwen3"}'


def test_tokenizer_fingerprint_is_model_path_independent():
    first = _Tokenizer()
    second = _Tokenizer()
    first.name_or_path = "/models/Qwen3-1.7B"
    second.name_or_path = "/models/Qwen3-8B"
    first.backend_tokenizer = _Backend()
    second.backend_tokenizer = _Backend()

    assert tokenizer_fingerprint(first) == tokenizer_fingerprint(second)


def _row(context, assistant=True):
    messages = [{"role": "user", "content": "question"}]
    if assistant:
        messages.append({"role": "assistant", "content": "answer"})
    return {"context": context, "conversations": messages}


def test_filtered_conversation_dataset_is_cached_and_invalidated(tmp_path: Path):
    source = tmp_path / "conversation.jsonl"
    source.write_text(
        "\n".join(json.dumps(row) for row in [_row("one two"), _row("three", False)]) + "\n",
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    tokenizer = _Tokenizer()

    first = ContextDataset(
        source,
        tokenizer=tokenizer,
        max_context_tokens=4,
        max_input_tokens=8,
        cache_dir=cache,
    )
    files_after_first = sorted(cache.glob("conversation_filtered-*.jsonl"))
    assert len(first) == 1
    assert len(files_after_first) == 1

    second = ContextDataset(
        source,
        tokenizer=tokenizer,
        max_context_tokens=4,
        max_input_tokens=8,
        cache_dir=cache,
    )
    assert len(second) == len(first)
    assert sorted(cache.glob("conversation_filtered-*.jsonl")) == files_after_first

    source.write_text(
        source.read_text(encoding="utf-8") + json.dumps(_row("four five six")) + "\n",
        encoding="utf-8",
    )
    third = ContextDataset(
        source,
        tokenizer=tokenizer,
        max_context_tokens=4,
        max_input_tokens=8,
        cache_dir=cache,
    )
    assert len(third) == 2
    assert len(list(cache.glob("conversation_filtered-*.jsonl"))) == 2


def test_batched_tokenizer_filter_respects_max_samples(tmp_path: Path):
    source = tmp_path / "conversation.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(_row(context))
            for context in ("one", "two", "three")
        ) + "\n",
        encoding="utf-8",
    )

    dataset = ContextDataset(
        source,
        tokenizer=_Tokenizer(),
        max_context_tokens=4,
        max_input_tokens=8,
        max_samples=1,
        tokenizer_filter_batch_size=2,
    )

    assert len(dataset) == 1


def test_question_and_answer_lengths_are_filtered_before_caching(tmp_path: Path):
    source = tmp_path / "conversation.jsonl"
    rows = [
        _row("short context"),
        {"context": "short context", "conversations": [
            {"role": "user", "content": "one two three four"},
            {"role": "assistant", "content": "answer"},
        ]},
        {"context": "short context", "conversations": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "one two three four"},
        ]},
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    dataset = ContextDataset(
        source,
        tokenizer=_Tokenizer(),
        max_context_tokens=4,
        max_input_tokens=8,
        max_question_tokens=3,
        max_answer_tokens=3,
        tokenizer_filter_batch_size=2,
    )

    assert len(dataset) == 1


def test_the_serial_and_parallel_filters_keep_the_same_rows(tmp_path: Path):
    """One filter implementation, two drivers: the rows must be identical.

    The rules used to be written twice (once for the process pool, once for the in-process
    path), which is exactly how a filter drifts between them -- the pool version stopped
    checking conversation length while the serial one still did, or the reverse.
    """
    source = tmp_path / "conversation.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(row)
            for row in [_row(f"one two three {index}") for index in range(6)]
            + [_row("one two", False), _row(""), _row("one" * 9)]
        )
        + "\n",
        encoding="utf-8",
    )

    def load(workers):
        return ContextDataset(
            source,
            tokenizer=_Tokenizer(),
            max_context_tokens=4,
            max_input_tokens=8,
            allow_empty=True,
            num_workers=workers,
        )

    serial, parallel = load(0), load(3)
    assert len(serial) == len(parallel)
    assert [record.context for record in serial] == [record.context for record in parallel]


def test_the_process_pool_filter_keeps_the_same_rows_in_the_same_order(tmp_path):
    """The loader's parallel filter and the cache build share one driver; order must hold.

    ``ContextDataset(num_workers=2)`` sends chunks through a process pool, and the dataset
    order decides which context a batch belongs to, so the accepted rows must come back exactly
    as the serial path returns them.
    """
    path = tmp_path / "rows.jsonl"
    rows = []
    for index in range(12):
        row = _row(f"context {index} " + "x" * index, assistant=index % 4 != 0)
        row["dataset"] = "squad"
        row["id"] = f"squad:{index}"
        rows.append(row)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    def contexts(num_workers):
        dataset = ContextDataset(
            path, tokenizer=_Tokenizer(), max_context_tokens=64, cache_dir=None,
            num_workers=num_workers,
        )
        return [record.context for record in dataset]

    serial = contexts(0)
    parallel = contexts(2)
    assert serial and serial == parallel
    assert len(serial) == 9, "the rows without an assistant turn are dropped in both modes"
