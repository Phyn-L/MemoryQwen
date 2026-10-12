import json
from pathlib import Path

from scripts.data.benchmark_jsonl_filter import benchmark


class _Tokenizer:
    def __call__(self, text, **kwargs):
        if isinstance(text, list):
            class BatchEncoded:
                input_ids = [item.split() for item in text]

            return BatchEncoded()

        class Encoded:
            input_ids = text.split()

        return Encoded()


def _row(context: str, assistant: bool = True):
    messages = [{"role": "user", "content": "question"}]
    if assistant:
        messages.append({"role": "assistant", "content": "answer"})
    return {"context": context, "conversations": messages}


def test_benchmark_reports_separate_parse_and_tokenizer_costs(tmp_path: Path):
    source = tmp_path / "conversation.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(row)
            for row in [_row("one two"), _row("one two three four five"), _row("x", False)]
        )
        + "\n",
        encoding="utf-8",
    )

    result = benchmark(
        source,
        tokenizer=_Tokenizer(),
        max_context_tokens=4,
        max_input_tokens=8,
        show_progress=False,
    )

    assert result["rows_seen"] == 3
    assert result["rows_accepted"] == 1
    assert result["rows_rejected_long_context"] == 1
    assert result["rows_rejected_no_assistant"] == 1
    timing = result["timing_seconds"]
    assert timing["wall"] >= timing["json_decode"]
    assert timing["wall"] >= timing["tokenizer_filter"]


def test_benchmark_max_rows_is_exact(tmp_path: Path):
    source = tmp_path / "conversation.jsonl"
    source.write_text(
        "\n".join(json.dumps(_row(f"row {index}")) for index in range(5)) + "\n",
        encoding="utf-8",
    )

    result = benchmark(
        source,
        tokenizer=_Tokenizer(),
        max_context_tokens=8,
        max_input_tokens=8,
        max_rows=2,
        show_progress=False,
    )

    assert result["rows_seen"] == 2
    assert result["rows_decoded"] == 2
