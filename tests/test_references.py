"""Every gold answer must survive the dataset and be used by the evaluator.

The evaluator used to score against the first reference only, while the ICL baseline took
the max over all of them. On SQuAD that is a systematic gap: of the 16498 answered QA pairs
in `aggregated/squad/validation.jsonl`, 12728 carry 3 references, 2092 carry 5 and 1384
carry 4 -- so the evaluator was stricter than the baseline it is compared against.

`first_token_em` had the same kind of quiet strictness: it compared raw token ids, so a
correct word encoded with a leading space (` cat`, what the model generates) never matched
the gold token (`cat`, what tokenizing the answer alone yields).
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


class _WordTokenizer:
    """Minimal stand-in: one token per word, and a leading space changes the id.

    That second property is the point. Real byte-level BPE encodes ``cat`` and `` cat``
    as different tokens for the same word, which is why the evaluator compares the
    normalized text of a single token as well as its id.
    """

    pad_token_id = 0

    def __init__(self):
        self.vocabulary: dict[str, int] = {}

    def _id(self, token: str) -> int:
        return self.vocabulary.setdefault(token, len(self.vocabulary) + 1)

    def __call__(self, text, add_special_tokens=False):
        if not text:
            return SimpleNamespace(input_ids=[])
        word = text.split()[0]
        return SimpleNamespace(input_ids=[self._id((" " if text[0].isspace() else "") + word)])

    def decode(self, ids, skip_special_tokens=True):
        for token, index in self.vocabulary.items():
            if index == int(ids[0]):
                return token
        return ""


def test_first_token_em_considers_every_reference():
    tokenizer = _WordTokenizer()
    evaluator = Evaluator(tokenizer, SimpleNamespace())
    assert evaluator._first_token_ids(QARecord("q", "cat", answers=("cat", "dog", ""))) == {
        tokenizer("cat").input_ids[0], tokenizer("dog").input_ids[0],
    }
    assert evaluator._first_token_ids(QARecord("q", "cat")) == {tokenizer("cat").input_ids[0]}


def test_first_token_em_sees_the_space_prefixed_generation():
    """The regression: gold ``cat`` vs generated `` cat`` must be a hit.

    A raw token-id comparison reports a miss for every correct answer whose first word is
    not at the start of a string, which is most of them once the model generates after the
    prompt's ``Answer:``.
    """
    tokenizer = _WordTokenizer()
    evaluator = Evaluator(tokenizer, SimpleNamespace())
    record = QARecord("q", "cat", answers=("cat", "dog"))
    generated = tokenizer(" cat").input_ids[0]
    assert generated != tokenizer("cat").input_ids[0], "the fake tokenizer must tell them apart"
    assert evaluator._first_token_hit(generated, record) is True
    assert evaluator._first_token_hit(tokenizer("cat").input_ids[0], record) is True
    assert evaluator._first_token_hit(tokenizer(" DOG").input_ids[0], record) is True


def test_first_token_em_rejects_wrong_empty_and_padding_tokens():
    tokenizer = _WordTokenizer()
    evaluator = Evaluator(tokenizer, SimpleNamespace())
    record = QARecord("q", "cat")
    assert evaluator._first_token_hit(tokenizer("bird").input_ids[0], record) is False
    assert evaluator._first_token_hit(tokenizer.pad_token_id, record) is False
    # An id the tokenizer cannot decode carries no text, so it cannot match by text either.
    assert evaluator._first_token_hit(max(tokenizer.vocabulary.values()) + 1, record) is False


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
