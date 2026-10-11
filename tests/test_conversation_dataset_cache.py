import json
from pathlib import Path

from src.cache_utils import tokenizer_fingerprint
from src.data import ConversationDataset


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


def test_conversation_offsets_are_cached_and_invalidated(tmp_path: Path):
    source = tmp_path / "conversation.jsonl"
    source.write_text(
        "\n".join(json.dumps(row) for row in [_row("one two"), _row("three", False)]) + "\n",
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    tokenizer = _Tokenizer()

    first = ConversationDataset(
        source,
        tokenizer=tokenizer,
        max_context_tokens=4,
        max_input_context_length=8,
        cache_dir=cache,
    )
    files_after_first = sorted(cache.glob("conversation_offsets-*.json"))
    assert len(first) == 1
    assert len(files_after_first) == 1

    second = ConversationDataset(
        source,
        tokenizer=tokenizer,
        max_context_tokens=4,
        max_input_context_length=8,
        cache_dir=cache,
    )
    assert len(second) == len(first)
    assert sorted(cache.glob("conversation_offsets-*.json")) == files_after_first

    source.write_text(
        source.read_text(encoding="utf-8") + json.dumps(_row("four five six")) + "\n",
        encoding="utf-8",
    )
    third = ConversationDataset(
        source,
        tokenizer=tokenizer,
        max_context_tokens=4,
        max_input_context_length=8,
        cache_dir=cache,
    )
    assert len(third) == 2
    assert len(list(cache.glob("conversation_offsets-*.json"))) == 2


def test_batched_tokenizer_filter_respects_max_samples(tmp_path: Path):
    source = tmp_path / "conversation.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(_row(context))
            for context in ("one", "two", "three")
        ) + "\n",
        encoding="utf-8",
    )

    dataset = ConversationDataset(
        source,
        tokenizer=_Tokenizer(),
        max_context_tokens=4,
        max_input_context_length=8,
        max_samples=1,
        tokenizer_filter_batch_size=2,
    )

    assert len(dataset) == 1
