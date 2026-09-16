"""Every gold answer must survive the dataset and be used by the evaluator.

The evaluator used to score against the first reference only, while the ICL baseline took
the max over all of them. On SQuAD that is a systematic gap: of the 16498 answered QA pairs
in `aggregated/squad/validation.jsonl`, 12728 carry 3 references, 2092 carry 5 and 1384
carry 4 -- so the evaluator was stricter than the baseline it is compared against.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import QARecord, _answers  # noqa: E402
from src.evaluator import Evaluator  # noqa: E402
from src.metrics import best_reference_metrics  # noqa: E402

QWEN_1_7B = (
    "/data/lz/hf_cache/hub/models--Qwen--Qwen3-1.7B/"
    "snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
)


def test_answers_keeps_every_non_empty_gold_answer():
    assert _answers({"answers": ["cat", "the cat", " a cat ", ""]}) == ("cat", "the cat", "a cat")
    assert _answers({"answer": "x"}) == ("x",)
    assert _answers({"answers": []}) == ()
    assert _answers({"answers": None}) == ()
    assert _answers({}) == ()


def test_first_answer_is_still_the_training_target():
    record = QARecord("q", "first", answers=("first", "second"))
    assert record.answer == "first", "training supervises one answer; it must be the first"
    assert record.references == ("first", "second")


def test_references_fall_back_to_the_single_answer():
    assert QARecord("q", "only").references == ("only",)
    assert QARecord("q", "").references == ()


class _FirstCharTokenizer:
    """Minimal stand-in: one 'token' per string, equal to its first character."""

    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(text[0])] if text else [])


def test_first_token_em_considers_every_reference():
    evaluator = Evaluator(_FirstCharTokenizer(), SimpleNamespace())
    assert evaluator._first_token_ids(QARecord("q", "cat", answers=("cat", "dog", ""))) == {
        ord("c"), ord("d"),
    }
    assert evaluator._first_token_ids(QARecord("q", "cat")) == {ord("c")}


def test_multi_reference_reduction_is_a_max_not_the_first_reference():
    prediction, references = "a dog", ("cat", "a dog", "the dog")
    scored = best_reference_metrics(prediction, references)
    first_only = best_reference_metrics(prediction, references[:1])
    assert scored["em"] == 1.0 and scored["f1"] == 1.0
    assert first_only["em"] == 0.0, "this is the stricter number the evaluator used to report"
    # A record carrying only the first answer reproduces that old, stricter number.
    assert best_reference_metrics(prediction, QARecord("q", "cat").references)["em"] == 0.0


def test_dataset_cache_roundtrip_keeps_every_reference():
    """The end-to-end path: jsonl -> dataset -> HF cache -> reload.

    This is the regression that matters. Adding a field to the cached schema without
    bumping ``dataset_cache.CACHE_VERSION`` would silently keep serving caches that lack
    it, so this test also fails if the cache is reused after the schema changed.
    """
    if not Path(QWEN_1_7B).exists():
        print("   (skipped: Qwen1.7B tokenizer not available)")
        return

    from transformers import AutoTokenizer

    from src.data import AggregatedContextDataset

    tokenizer = AutoTokenizer.from_pretrained(QWEN_1_7B, local_files_only=True)
    with tempfile.TemporaryDirectory() as root:
        dataset_dir = Path(root) / "squad"
        dataset_dir.mkdir()
        rows = [
            {"context_id": "c1", "context": "Paris is in France.",
             "qa_pairs": [{"question": "Where?", "answers": ["France", "the country of France"]}]},
            {"context_id": "c2", "context": "The cat sat on the mat.",
             "qa_pairs": [{"question": "Who sat?", "answers": ["cat", "the cat"]}]},
        ]
        (dataset_dir / "train.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8",
        )

        cache_dir = Path(root) / "outputs"
        built = AggregatedContextDataset(root, "all", "train", tokenizer, 64, cache_dir=cache_dir)
        reloaded = AggregatedContextDataset(root, "all", "train", tokenizer, 64, cache_dir=cache_dir)

        for name, dataset in (("built", built), ("reloaded from cache", reloaded)):
            collected = [pair for record in dataset for pair in record.qa_pairs]
            assert len(collected) == 2, name
            assert [pair.references for pair in collected] == [
                ("France", "the country of France"), ("cat", "the cat"),
            ], name
            assert [pair.answer for pair in collected] == ["France", "cat"], name


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
