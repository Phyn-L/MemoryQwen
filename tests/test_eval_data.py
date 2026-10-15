"""One place decides where a dataset's evaluation file lives.

``src/eval_data.py`` exists because two suites (the memory-side orchestrator and the ICL
baseline) plus ``scripts/test.py`` all have to agree on the layout ``scripts/data/`` writes;
they used to read the deprecated ``aggregated`` tree and drifted apart from the training
pipeline. These tests pin the layout, the JSONL dialects and the demo sampler.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval_data import (
    DATASETS,
    DEFAULT_DATASETS,
    meta_file,
    report_name,
    resolve,
    slug,
    split_file,
    train_file,
    validation_file,
)
from src.icl_baseline import iter_examples, load_jsonl, sample_jsonl_many


def _write(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _conversation_row(context="a b c", question="q?", answer="a", dataset="squad"):
    return {
        "context": context,
        "dataset": dataset,
        "id": f"{dataset}:x",
        "conversations": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer, "answers": [answer]},
        ],
    }


def _pair_row(context="c", pairs=(("q1", "a1"), ("q2", "a2")), dataset="squad", **metadata):
    """One training-split row. RACE needs ``answer_letter`` metadata to be usable as a demo."""
    return {
        "context": context,
        "dataset": dataset,
        "id": f"{dataset}:y",
        "metadata": dict(metadata),
        "qa_pairs": [
            {
                "id": f"{dataset}:y:{index}",
                "question": q,
                "answers": [a],
                "answer": a,
                "metadata": dict(metadata),
            }
            for index, (q, a) in enumerate(pairs)
        ],
    }


# --------------------------------------------------------------------------- layout


def test_every_shipped_dataset_has_a_slug_and_a_default_subset():
    assert set(DEFAULT_DATASETS) <= set(DATASETS)
    assert slug("hotpotqa(distractor)") == "hotpotqa_distractor"
    assert slug("msmarcov1.1") == "msmarcov1.1"
    with pytest.raises(ValueError, match="unknown dataset"):
        slug("ms_marco")


def test_split_files_live_under_the_data_root():
    # Raw splits live under meta/ (the builder's output) ...
    assert (
        str(split_file("contexts", "squad", "test")) == "contexts/1qa/meta/test/squad.json"
    )
    assert (
        str(split_file("contexts", "race", "validation"))
        == "contexts/1qa/meta/validation/race.json"
    )
    assert str(meta_file("contexts", "test")) == "contexts/1qa/meta/meta_test.json"
    # ... and the files a run reads are the filtered cache.
    assert str(train_file("contexts")) == "contexts/1qa/cache/train.jsonl"
    assert str(validation_file("contexts")) == "contexts/1qa/cache/validation.jsonl"
    assert report_name("hotpotqa(distractor)") == "hotpotqa_distractor.json"
    with pytest.raises(ValueError, match="unknown evaluation split"):
        split_file("contexts", "squad", "train")


def test_resolve_prefers_the_test_split_then_falls_back(tmp_path):
    _write(split_file(tmp_path, "squad", "validation"), [_conversation_row()])
    path, split = resolve(tmp_path, "squad")
    assert (Path(path), split) == (split_file(tmp_path, "squad", "validation"), "validation")

    _write(split_file(tmp_path, "squad", "test"), [_conversation_row()])
    path, split = resolve(tmp_path, "squad")
    assert (Path(path), split) == (split_file(tmp_path, "squad", "test"), "test")

    # An empty file is not a split.
    split_file(tmp_path, "race", "test").parent.mkdir(parents=True, exist_ok=True)
    split_file(tmp_path, "race", "test").write_text("")
    with pytest.raises(FileNotFoundError, match="no nonempty"):
        resolve(tmp_path, "race")


def test_resolve_names_the_builder_that_writes_the_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_1qa.py"):
        resolve(tmp_path, "squad")


# --------------------------------------------------------------------------- readers


def test_the_conversation_dialect_yields_one_example_per_assistant_answer(tmp_path):
    path = _write(tmp_path / "squad.jsonl", [_conversation_row()])
    rows = list(iter_examples(path, "squad"))
    assert len(rows) == 1
    flat, line_number, index = rows[0]
    assert (line_number, index) == (1, 0)
    assert flat["question"] == "q?"
    assert flat["answers"] == ["a"]
    assert flat["context"] == "a b c"
    examples = load_jsonl(path, "squad")
    assert [example.question for example in examples] == ["q?"]


def test_the_qa_pairs_dialect_of_the_training_split_is_readable(tmp_path):
    path = _write(tmp_path / "train.jsonl", [_pair_row()])
    rows = list(iter_examples(path, "squad"))
    assert [row[0]["question"] for row in rows] == ["q1", "q2"]
    assert [row[0]["id"] for row in rows] == ["squad:y:0", "squad:y:1"]


def test_a_shared_file_is_filtered_by_dataset(tmp_path):
    path = _write(
        tmp_path / "train.jsonl",
        [_pair_row(dataset="squad"), _pair_row(dataset="race", pairs=(("rq", "ra"),))],
    )
    assert [row[0]["question"] for row in iter_examples(path, "squad")] == ["q1", "q2"]
    assert [row[0]["question"] for row in iter_examples(path, "race")] == ["rq"]
    assert len(list(iter_examples(path, None))) == 3


def test_rows_without_a_gold_answer_are_skipped(tmp_path):
    path = _write(
        tmp_path / "squad.jsonl",
        [
            {
                "context": "c",
                "dataset": "squad",
                "conversations": [{"role": "user", "content": "q?"}],
            }
        ],
    )
    assert load_jsonl(path, "squad") == []


def test_a_file_with_neither_dialect_fails_loudly(tmp_path):
    path = _write(tmp_path / "squad.jsonl", [{"context": "c", "dataset": "squad"}])
    with pytest.raises(ValueError, match="neither 'conversations' nor 'qa_pairs'"):
        list(iter_examples(path, "squad"))


# --------------------------------------------------------------------------- demos


def test_demo_sampling_is_one_pass_and_per_dataset(tmp_path):
    path = _write(
        tmp_path / "train.jsonl",
        [
            _pair_row(dataset="squad", pairs=((f"q{i}", f"a{i}") for i in range(4))),
            _pair_row(
                dataset="race", pairs=((f"r{i}", f"b{i}") for i in range(4)),
                answer_letter="A",
            ),
        ],
    )
    sampled = sample_jsonl_many(path, ["squad", "race"], 2, seed=7)
    assert sorted(sampled) == ["race", "squad"]
    assert all(len(rows) == 2 for rows in sampled.values())
    assert {example.dataset for example in sampled["squad"]} == {"squad"}
    assert {example.dataset for example in sampled["race"]} == {"race"}


def test_demo_sampling_is_reproducible_and_independent_of_the_requested_set(tmp_path):
    path = _write(
        tmp_path / "train.jsonl",
        [_pair_row(dataset="squad", pairs=((f"q{i}", f"a{i}") for i in range(6))),
         _pair_row(dataset="race", pairs=((f"r{i}", f"b{i}") for i in range(6)),
                   answer_letter="B")],
    )
    alone = sample_jsonl_many(path, ["squad"], 2, seed=11)["squad"]
    together = sample_jsonl_many(path, ["race", "squad"], 2, seed=11)["squad"]
    repeat = sample_jsonl_many(path, ["squad"], 2, seed=11)["squad"]
    assert [example.id for example in alone] == [example.id for example in together]
    assert [example.id for example in alone] == [example.id for example in repeat]
    other_seed = sample_jsonl_many(path, ["squad"], 2, seed=12)["squad"]
    assert [example.id for example in other_seed] != [example.id for example in alone]


def test_zero_shot_sampling_does_not_open_the_file(tmp_path):
    # count == 0 is the common case (zero-shot ICL) and the train split is several GB.
    assert sample_jsonl_many(tmp_path / "does-not-exist.jsonl", ["squad"], 0) == {"squad": []}


def test_a_short_demo_pool_is_an_error_not_a_silent_smaller_shot_count(tmp_path):
    path = _write(tmp_path / "train.jsonl", [_pair_row(dataset="squad", pairs=(("q", "a"),))])
    with pytest.raises(ValueError, match="found fewer for"):
        sample_jsonl_many(path, ["squad"], 4)


def test_an_older_data_tree_still_resolves(tmp_path):
    """A tree built by the previous layout keeps working until it is rebuilt."""
    from src.eval_data import FALLBACK_SPLIT_DIRS, split_locations

    assert FALLBACK_SPLIT_DIRS["test"] == (("1qa-test", ".jsonl"),)
    assert FALLBACK_SPLIT_DIRS["validation"] == (
        ("1qa-validation", ".jsonl"), ("1qa-val", ".jsonl"),
    )
    assert split_locations("test")[0] == ("meta/test", ".json")
    sampled = tmp_path / "1qa" / "1qa-val" / "squad.jsonl"
    _write(sampled, [_conversation_row()])
    path, split = resolve(tmp_path, "squad", "validation")
    assert (Path(path), split) == (sampled, "validation")
