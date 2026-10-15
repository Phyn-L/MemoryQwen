"""The 1QA builder: source conventions, and the layout the readers expect.

The corpus has three artifacts and a small set of conventions (``contexts/README.md``):
one line per context with all of its QA pairs, no filtering beyond dropping structurally empty
samples, MS MARCO contexts built from the selected passages only, and the per-dataset
evaluation files under the directory names ``src/eval_data.py`` owns.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.data.build_1qa import (  # noqa: E402
    BUILD_DATASETS,
    build,
    convert_rows,
    write_corpus,
)
from src.eval_data import meta_file, split_file, train_file, validation_file  # noqa: E402


def _write_jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return path


# ------------------------------------------------------------------ per-dataset conventions
def test_msmarco_context_is_built_from_the_selected_passages_only():
    row = {
        "query": "when was it founded ?",
        "answers": ["1901"],
        "query_type": "NUMERIC",
        "passages": {
            "passage_text": ["UNRELATED distractor text", "the selected passage", "also selected"],
            "is_selected": [0, 1, 1],
        },
    }
    converted, = convert_rows("msmarcov1.1", "train", [row])
    assert converted["context"] == "the selected passage\n\nalso selected"
    assert converted["dataset"] == "msmarcov1.1"
    assert converted["source_split"] == "train"
    pair, = converted["qa_pairs"]
    assert pair["question"] == "when was it founded ?"
    assert pair["answers"] == ["1901"]


def test_msmarco_rows_without_a_selected_passage_are_meaningless_and_dropped():
    row = {
        "query": "q",
        "answers": ["a"],
        "passages": {"passage_text": ["only a distractor"], "is_selected": [0]},
    }
    assert convert_rows("msmarcov1.1", "train", [row]) == []


def test_squad_keeps_every_reference_and_drops_the_unanswerable_pairs():
    rows = [
        {
            "id": "1", "title": "T", "context": "Paris is in France.",
            "question": "Where?", "answers": {"text": ["France", "France"], "answer_start": [10, 10]},
        },
        {
            "id": "2", "title": "T", "context": "Paris is in France.",
            "question": "Unanswerable?", "answers": {"text": [], "answer_start": []},
        },
    ]
    converted, = convert_rows("squad", "validation", rows)
    assert converted["context"] == "Paris is in France."
    pair, = converted["qa_pairs"]
    assert pair["answers"] == ["France"], "duplicates collapse, but aliases are kept"


def test_all_questions_of_one_context_are_aggregated_in_file_order():
    rows = [
        {"context": "same context", "question": "q2", "answers": {"text": ["a2"]}},
        {"context": "other context", "question": "q3", "answers": {"text": ["a3"]}},
        {"context": "same context", "question": "q1", "answers": {"text": ["a1"]}},
    ]
    first, second = convert_rows("squad", "train", rows)
    assert [row["context"] for row in (first, second)] == ["same context", "other context"]
    assert [pair["question"] for pair in first["qa_pairs"]] == ["q2", "q1"]
    assert len(first["qa_pairs"]) == 2 and len(second["qa_pairs"]) == 1
    # One id per context row, plus the pair ids.
    assert first["id"].startswith("squad:train:")


def test_race_keeps_the_options_and_the_answer_letter():
    row = {
        "article": "A short passage.", "question": "Which?",
        "options": ["one", "two", "three", "four"], "answer": "C",
    }
    converted, = convert_rows("race", "test", [row])
    pair, = converted["qa_pairs"]
    assert pair["answers"] == ["three"]
    assert pair["metadata"] == {"options": ["one", "two", "three", "four"], "answer_letter": "C"}


def test_a_race_row_whose_letter_does_not_match_the_options_is_dropped():
    row = {"article": "p", "question": "q", "options": ["one"], "answer": "C"}
    assert convert_rows("race", "test", [row]) == []


def test_hotpotqa_context_is_the_rows_own_paragraphs():
    row = {
        "context": {"title": ["A", "B"], "sentences": [["first.", "second."], ["third."]]},
        "question": "q", "answer": "a", "type": "comparison", "level": "hard",
        "supporting_facts": {"title": ["A"], "sent_id": [0]},
    }
    converted, = convert_rows("hotpotqa(distractor)", "validation", [row])
    assert converted["context"] == "A: first.second.\n\nB: third."
    pair, = converted["qa_pairs"]
    assert pair["metadata"]["level"] == "hard"


def test_only_structurally_empty_samples_are_dropped_no_length_caps():
    """A huge context and a long question survive; blank ones do not."""
    long_context = " ".join(f"token{index}" for index in range(5000))
    long_question = " ".join(f"w{index}" for index in range(400))
    rows = [
        {"context": long_context, "question": long_question, "answers": {"text": ["a"]}},
        {"context": "   ", "question": "q", "answers": {"text": ["a"]}},
        {"context": "c", "question": "  ", "answers": {"text": ["a"]}},
        {"context": "c", "question": "q", "answers": {"text": ["", "  "]}},
    ]
    converted = convert_rows("squad", "train", rows)
    assert len(converted) == 1
    assert converted[0]["context"] == long_context
    assert len(converted[0]["qa_pairs"][0]["question"]) == len(long_question)


def test_musique_keeps_the_answer_aliases_as_extra_references():
    row = {
        "question": "q",
        "answer": "the answer",
        "answer_aliases": ["alias one", ""],
        "paragraphs": [{"paragraph_text": "first paragraph"}, {"paragraph_text": "second"}],
    }
    converted, = convert_rows("musique", "validation", [row])
    assert converted["context"] == "first paragraph\n\nsecond"
    assert converted["qa_pairs"][0]["answers"] == ["the answer", "alias one"]


# ----------------------------------------------------------------------------------- layout
def test_write_corpus_writes_the_layout_the_readers_expect(tmp_path):
    rows_by_split = {
        "train": {"squad": [{"id": "a", "dataset": "squad", "source_split": "train",
                             "context": "c", "qa_pairs": [{"id": "p", "question": "q",
                                                           "answers": ["a"], "answer": "a",
                                                           "metadata": {}}],
                             "metadata": {}}]},
        "validation": {"race": [{"id": "b", "dataset": "race", "source_split": "validation",
                                 "context": "c2", "qa_pairs": [{"id": "p2", "question": "q2",
                                                                "answers": ["a2"], "answer": "a2",
                                                                "metadata": {}}],
                                 "metadata": {}}]},
    }
    # The builder's root IS the 1qa directory; raw rows land under its meta/ tree.
    root = tmp_path / "1qa"
    manifest = write_corpus(root, rows_by_split)

    assert (root / "meta" / "meta_train.json").is_file()
    assert (root / "meta" / "meta_validation.json").is_file()
    assert not (root / "meta" / "meta_test.json").exists(), "an empty split is not written"
    assert split_file(tmp_path, "race", "validation") == root / "meta" / "validation" / "race.json"
    assert split_file(tmp_path, "race", "validation").is_file()
    assert meta_file(tmp_path, "train") == root / "meta" / "meta_train.json"
    # The filtered cache is a different script's output; the reader names it regardless.
    assert train_file(tmp_path) == root / "cache" / "train.jsonl"
    assert validation_file(tmp_path) == root / "cache" / "validation.jsonl"
    assert not train_file(tmp_path).exists()
    assert manifest["splits"]["train"]["contexts"] == 1
    assert manifest["splits"]["train"]["datasets"]["squad"]["qa_pairs"] == 1
    assert manifest["missing_splits"]["test"] == list(BUILD_DATASETS)
    assert (root / "meta" / "build.manifest.json").is_file()


def _fake_cache(tmp_path) -> Path:
    """A miniature HF cache: one parquet dataset, the legacy SQuAD arrow stream, one jsonl."""
    pyarrow = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    cache = tmp_path / "cache"
    snapshot = cache / "hub" / "datasets--hotpotqa--hotpot_qa" / "snapshots" / "rev"
    (snapshot / "distractor").mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "context": {"title": [f"A{index}"], "sentences": [[f"text {index}."]]},
            "question": f"q{index}", "answer": f"a{index}", "type": "bridge", "level": "easy",
            "supporting_facts": {"title": [f"A{index}"], "sent_id": [0]},
        }
        for index in range(3)
    ]
    pq.write_table(
        pyarrow.Table.from_pylist(rows),
        snapshot / "distractor" / "validation-00000-of-00001.parquet",
    )

    squad = cache / "datasets/squad/plain_text/0.0.0/hash"
    squad.mkdir(parents=True, exist_ok=True)
    schema = pyarrow.schema([
        ("id", pyarrow.string()), ("title", pyarrow.string()), ("context", pyarrow.string()),
        ("question", pyarrow.string()),
        ("answers", pyarrow.struct([("text", pyarrow.list_(pyarrow.string())),
                                    ("answer_start", pyarrow.list_(pyarrow.int32()))])),
    ])
    with pyarrow.ipc.new_stream(str(squad / "squad-validation.arrow"), schema) as stream:
        stream.write_batch(pyarrow.record_batch(
            {"id": ["1"], "title": ["T"], "context": ["c"], "question": ["q"],
             "answers": [{"text": ["a"], "answer_start": [0]}]},
            schema=schema,
        ))

    musique = cache / "hub" / "datasets--dgslibisey--MuSiQue" / "snapshots" / "rev"
    musique.mkdir(parents=True, exist_ok=True)
    (musique / "musique_ans_v1.0_dev.jsonl").write_text(
        json.dumps({"question": "q", "answer": "a", "paragraphs": [{"paragraph_text": "p"}]}) + "\n",
        encoding="utf-8",
    )
    return cache


def test_the_builder_reads_a_local_cache_end_to_end(tmp_path):
    cache = _fake_cache(tmp_path)
    manifest = build(
        cache, tmp_path / "out",
        ["squad", "hotpotqa(distractor)", "musique", "race"], ["validation"],
        quiet=True,
    )
    assert set(manifest["splits"]["validation"]["datasets"]) == {"squad", "hotpotqa(distractor)", "musique"}
    assert "race" in manifest["missing_splits"]["validation"]
    merged = (
        tmp_path / "out" / "meta" / "meta_validation.json"
    ).read_text(encoding="utf-8").splitlines()
    # One squad row + three hotpotqa rows (.json per dataset) + one musique row.
    assert len(merged) == 5
    assert (tmp_path / "out" / "meta" / "validation" / "hotpotqa_distractor.json").is_file()


def test_a_parallel_build_writes_the_same_bytes_as_a_serial_one(tmp_path):
    """The pool returns tasks in dataset order, so the merged file cannot be reordered."""
    cache = _fake_cache(tmp_path)
    datasets = ["squad", "hotpotqa(distractor)", "musique"]
    serial_root, parallel_root = tmp_path / "serial", tmp_path / "parallel"

    serial = build(cache, serial_root, datasets, ["validation"], quiet=True, num_workers=1)
    parallel = build(cache, parallel_root, datasets, ["validation"], quiet=True, num_workers=4)

    assert serial["splits"] == parallel["splits"], "same counts, same order"
    files = sorted(
        path.relative_to(serial_root) for path in serial_root.rglob("*") if path.is_file()
    )
    assert files, "the build wrote something"
    for relative in files:
        assert (serial_root / relative).read_bytes() == (parallel_root / relative).read_bytes(), (
            f"{relative} differs between the serial and the parallel build"
        )


def test_a_single_worker_stays_in_process(tmp_path):
    """`--num-workers 1` must not pay for a pool (it is the debugging/CI path)."""
    cache = _fake_cache(tmp_path)
    manifest = build(
        cache, tmp_path / "out", ["squad"], ["validation"], quiet=True, num_workers=1
    )
    assert manifest["splits"]["validation"]["datasets"]["squad"]["contexts"] == 1
