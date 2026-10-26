"""Every gold answer must survive the dataset and be used by the evaluator.

The evaluator scores against every reference. On SQuAD that matters because many answered QA
pairs carry three or more references (the counts per file are in
`contexts/1qa/1qa-clean-splits.manifest.json`).

`first_token_em` had the same kind of quiet strictness: it compared raw token ids, so a
correct word encoded with a leading space (` cat`, what the model generates) never matched
the gold token (`cat`, what tokenizing the answer alone yields).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.records import _messages_from_row  # noqa: E402
from src.evaluation import Evaluator  # noqa: E402
from src.evaluation.eval_metrics import best_reference_metrics  # noqa: E402


def test_messages_keep_every_non_empty_gold_answer():
    messages = _messages_from_row({
        "conversations": [{
            "role": "assistant",
            "content": "cat",
            "answers": ["cat", "the cat", " a cat ", ""],
        }],
    })
    assert messages[0].answers == ("cat", "the cat", "a cat")


def test_race_row_uses_the_prebuilt_text_question_and_answer():
    row = {
        "dataset": "race",
        "qa_pairs": [{
            "question": "Which option?",
            "answers": ["option text"],
            "answer": "option text",
            "metadata": {"options": ["first", "option text", "third", "fourth"]},
        }],
    }
    user, assistant = _messages_from_row(row)
    assert user.content == (
        "Which option?\nOptions:\nfirst\noption text\nthird\nfourth"
    )
    assert assistant.content == "option text"
    assert assistant.answers == ("option text",)
    assert best_reference_metrics("option text", assistant.answers) == {
        "em": 1.0,
        "f1": 1.0,
        "rouge_l": 1.0,
    }
    assert best_reference_metrics("unrelated", assistant.answers) == {
        "em": 0.0,
        "f1": 0.0,
        "rouge_l": 0.0,
    }


def test_message_references_fall_back_to_the_single_answer():
    messages = _messages_from_row({
        "conversations": [{"role": "assistant", "content": "only"}],
    })
    assert messages[0].answers == ()


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
    record = SimpleNamespace(references=("cat", "dog", ""))
    assert evaluator._first_token_ids(record) == {
        tokenizer("cat").input_ids[0], tokenizer("dog").input_ids[0],
    }
    assert evaluator._first_token_ids(SimpleNamespace(references=("cat",))) == {
        tokenizer("cat").input_ids[0],
    }


def test_first_token_em_sees_the_space_prefixed_generation():
    """The regression: gold ``cat`` vs generated `` cat`` must be a hit.

    A raw token-id comparison reports a miss for every correct answer whose first word is
    not at the start of a string, which is most of them once the model generates after the
    prompt's ``Answer:``.
    """
    tokenizer = _WordTokenizer()
    evaluator = Evaluator(tokenizer, SimpleNamespace())
    record = SimpleNamespace(references=("cat", "dog"))
    generated = tokenizer(" cat").input_ids[0]
    assert generated != tokenizer("cat").input_ids[0], "the fake tokenizer must tell them apart"
    assert evaluator._first_token_hit(generated, record) is True
    assert evaluator._first_token_hit(tokenizer("cat").input_ids[0], record) is True
    assert evaluator._first_token_hit(tokenizer(" DOG").input_ids[0], record) is True


def test_first_token_em_rejects_wrong_empty_and_padding_tokens():
    tokenizer = _WordTokenizer()
    evaluator = Evaluator(tokenizer, SimpleNamespace())
    record = SimpleNamespace(references=("cat",))
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


def test_conversation_dataset_offset_cache_keeps_every_reference(tmp_path):
    """The JSONL offset cache must preserve every answer reference."""
    from src.data.dataset import ContextDataset

    class _Tokenizer:
        name_or_path = "reference-test-tokenizer"
        vocab_size = 32

        def __call__(self, texts, **kwargs):
            if isinstance(texts, str):
                texts = [texts]

            class Encoded:
                input_ids = [text.split() for text in texts]

            return Encoded()

    source = tmp_path / "conversation.jsonl"
    source.write_text(
        json.dumps({
            "context": "Paris is in France.",
            "conversations": [{
                "role": "user",
                "content": "Where?",
            }, {
                "role": "assistant",
                "content": "France",
                "answers": ["France", "the country of France"],
            }],
        }) + "\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache"
    tokenizer = _Tokenizer()

    built = ContextDataset(
        source,
        tokenizer=tokenizer,
        max_context_tokens=64,
        cache_dir=cache_dir,
    )
    reloaded = ContextDataset(
        source,
        tokenizer=tokenizer,
        max_context_tokens=64,
        cache_dir=cache_dir,
    )

    for dataset in (built, reloaded):
        assert dataset[0].conversations[-1].answers == (
            "France",
            "the country of France",
        )


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
