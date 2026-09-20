"""The ICL baseline and the training evaluator must read the same questions.

`scripts/evaluation/test_icl_baseline.py` used to default to `contexts/standardized/` while the
training pipeline read `contexts/aggregated/`, and the two files do not contain the same
rows (10570 vs 16498 answered), so the two means were over differently weighted sets. The
aggregated tree is now the only supported schema -- `iter_examples` raises on anything else
-- and the parity is asserted here instead of being assumed.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data import AggregatedContextDataset  # noqa: E402
from src.icl_baseline import iter_examples, load_jsonl  # noqa: E402
from utils.config import TrainConfig  # noqa: E402

CONFIG = REPO / "configs/train_baseline.yaml"


def _key(context: str, question: str) -> tuple[str, str]:
    """Whitespace/case-insensitive identity, so formatting cannot fake a difference."""
    return (" ".join(context.split()).lower(), " ".join(question.split()).lower())


def _write(path: Path, rows) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_a_non_aggregated_file_raises_instead_of_looking_empty():
    """A one-question-per-line file must fail loudly, not yield nothing.

    Pointing --squad-validation-file at the standardized tree silently produced an empty
    evaluation set before; the aggregated schema is the only supported input now.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(Path(tmp) / "flat.jsonl", [
            {"id": "a", "context": "ctx", "question": "q", "answers": ["x"]},
        ])
        try:
            load_jsonl(path, "squad")
        except ValueError as error:
            assert "qa_pairs" in str(error) and "aggregated" in str(error), error
        else:
            raise AssertionError("a flat/standardized file must be rejected")


def test_aggregated_schema_is_exploded_per_question():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(Path(tmp) / "agg.jsonl", [{
            "context": "the context", "context_id": "c1",
            "qa_pairs": [
                {"id": "q1", "question": "one?", "answers": ["a1", "a1b"]},
                {"id": "q2", "question": "two?", "answers": ["a2"]},
            ],
        }])
        examples = load_jsonl(path, "squad")
        assert [example.id for example in examples] == ["q1", "q2"]
        assert all(example.context == "the context" for example in examples)
        assert examples[0].references == ("a1", "a1b")


def test_questions_without_a_gold_answer_are_skipped():
    """The aggregated squad file carries SQuAD v2.0's unanswerable questions."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(Path(tmp) / "mixed.jsonl", [{
            "context": "ctx",
            "qa_pairs": [
                {"id": "answerable", "question": "q1", "answers": ["a"]},
                {"id": "impossible", "question": "q2", "answers": []},
                {"id": "blank", "question": "q3", "answers": ["  "]},
            ],
        }])
        assert [example.id for example in load_jsonl(path, "squad")] == ["answerable"]


def test_rows_without_ids_get_unique_identifiers():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(Path(tmp) / "noid.jsonl", [
            {"context": "c", "qa_pairs": [{"question": "n1", "answers": ["a"]}]},
            {"context": "c2", "qa_pairs": [
                {"question": "n2", "answers": ["a"]},
                {"question": "n3", "answers": ["b"]},
            ]},
        ])
        ids = [example.id for example in load_jsonl(path, "squad")]
        assert len(set(ids)) == 3, ids


def test_aggregated_race_rows_keep_their_options():
    """RACE needs `metadata.options`/`answer_letter`; the aggregation must preserve them."""
    root = TrainConfig.from_file(CONFIG).data.root
    path = Path(root) / "race" / "test.jsonl"
    if not path.exists():
        print("   (skipped: aggregated race data not available)")
        return
    examples = load_jsonl(path, "race")
    assert examples, "no RACE examples loaded"
    assert all(len(example.options) == 4 for example in examples), "options were lost"
    assert all(example.answer_letter in {"A", "B", "C", "D"} for example in examples)


def test_ms_marco_icl_uses_all_candidate_passages(tmp_path):
    import json
    from src.icl_baseline import load_jsonl
    path = tmp_path / "validation.jsonl"
    path.write_text(json.dumps({"context": "selected", "qa_pairs": [{
        "id": "m1", "question": "q", "answers": ["a"],
        "source_dataset": "ms_marco_v1_1",
        "metadata": {"passages": ["first", "second"], "is_selected": [1, 0]},
    }]}) + "\n")
    examples = load_jsonl(path, "ms_marco", "ms_marco_v1_1")
    assert examples[0].context == "first\n\nsecond"


def test_icl_default_file_and_training_validation_split_are_the_same_questions():
    """The invariant that makes the two headline numbers comparable."""
    config = TrainConfig.from_file(CONFIG)
    root = Path(config.data.root)
    icl_path = root / "squad" / "validation.jsonl"
    if not icl_path.exists():
        print("   (skipped: aggregated squad data not available)")
        return

    baseline = {
        _key(example.context, example.question)
        for example in load_jsonl(icl_path, "squad")
    }
    # Read the split the way training does, minus tokenization (filter_long_context=False
    # needs no tokenizer), so this compares question sets rather than filtering behaviour.
    dataset = AggregatedContextDataset(
        str(root), "squad", "validation", None, config.data.max_context_tokens,
        filter_long_context=False, filter_no_qa=True,
    )
    evaluator = {
        _key(record.context, pair.question)
        for record in dataset for pair in record.qa_pairs
    }

    assert baseline == evaluator, (
        f"ICL default and training validation disagree: "
        f"only-baseline={len(baseline - evaluator)}, only-evaluator={len(evaluator - baseline)}"
    )
    assert len(baseline) > 0


def test_iter_examples_reports_line_and_index():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(Path(tmp) / "agg.jsonl", [
            {"context": "c", "qa_pairs": [{"question": "n1", "answers": ["a"]}]},
            {"context": "c2", "qa_pairs": [{"question": "n2", "answers": ["a"]},
                                           {"question": "n3", "answers": ["b"]}]},
        ])
        seen = [
            (line_number, index, row["question"])
            for row, line_number, index in iter_examples(path, "squad")
        ]
        assert seen == [(1, 0, "n1"), (2, 0, "n2"), (2, 1, "n3")]

        # The flattened row keeps the context from the enclosing line.
        flat_row = next(iter(iter_examples(path, "squad")))[0]
        assert flat_row["context"] == "c"


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
            except Exception as error:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(error).__name__}: {error}")
            else:
                print(f"ok   {name}")
    print("\nVERDICT:", "ALL PASSED" if not failures else f"{failures} FAILED")
    raise SystemExit(1 if failures else 0)
