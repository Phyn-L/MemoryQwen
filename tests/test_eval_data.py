"""Dataset evaluation paths and the layout written by the 1QA builder."""

from pathlib import Path

import pytest

from src.evaluation.eval_data import (
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


def _write(path: Path, content="{}\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_every_shipped_dataset_has_a_slug_and_a_default_subset():
    assert set(DEFAULT_DATASETS) <= set(DATASETS)
    assert slug("hotpotqa(distractor)") == "hotpotqa_distractor"
    assert slug("msmarcov1.1") == "msmarcov1.1"
    with pytest.raises(ValueError, match="unknown dataset"):
        slug("ms_marco")


def test_split_files_live_under_the_data_root():
    assert str(split_file("contexts", "squad", "test")) == "contexts/1qa/meta/test/squad.json"
    assert (
        str(split_file("contexts", "race", "validation"))
        == "contexts/1qa/meta/validation/race.json"
    )
    assert str(meta_file("contexts", "test")) == "contexts/1qa/meta/meta_test.json"
    assert str(train_file("contexts")) == "contexts/1qa/cache/train.jsonl"
    assert str(validation_file("contexts")) == "contexts/1qa/cache/validation.jsonl"
    assert report_name("hotpotqa(distractor)") == "hotpotqa_distractor.json"
    with pytest.raises(ValueError, match="unknown evaluation split"):
        split_file("contexts", "squad", "train")


def test_resolve_prefers_test_then_falls_back_to_validation(tmp_path):
    validation = _write(split_file(tmp_path, "squad", "validation"))
    assert resolve(tmp_path, "squad") == (validation, "validation")

    test = _write(split_file(tmp_path, "squad", "test"))
    assert resolve(tmp_path, "squad") == (test, "test")

    _write(split_file(tmp_path, "race", "test"), "")
    with pytest.raises(FileNotFoundError, match="no nonempty"):
        resolve(tmp_path, "race")


def test_resolve_names_the_builder_that_writes_the_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_1qa.py"):
        resolve(tmp_path, "squad")


def test_only_canonical_data_tree_resolves(tmp_path):
    from src.evaluation.eval_data import split_locations

    assert split_locations("test") == (("meta/test", ".json"),)
