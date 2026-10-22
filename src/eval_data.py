"""Where a dataset's evaluation file lives in the layout ``scripts/data/`` builds.

One module decides this, so the memory-side orchestrator
(``scripts/evaluation/test_all_suite.py``) and the ICL baseline
(``scripts/evaluation/test_icl_suite.py``) cannot disagree about the paths again. They used to
read ``<root>/<dataset>/{train,validation,test}.jsonl``, the layout of the deprecated
``aggregated`` corpus, while the training pipeline moved to the 1QA build::

    <root>/1qa/meta/<split>/<slug>.json      # the dataset's own raw rows
    <root>/1qa/meta/meta_<split>.json        # the same rows for every dataset, merged
    <root>/1qa/cache/train.jsonl             # filtered training split (from meta_train)
    <root>/1qa/cache/validation.jsonl        # filtered validation split (from meta_validation)

The raw splits come from ``scripts/data/build_1qa.py`` (local Hugging Face cache, unfiltered),
the filtered ``cache/`` files from ``scripts/data/build_cache.py`` (this project's YAML filters).
"""
from __future__ import annotations

from pathlib import Path

#: Evaluation splits, and the directory ``scripts/data/build_1qa.py`` writes each one into.
SPLIT_DIRS = {"test": "test", "validation": "validation"}

#: Datasets the reader knows, mapped to the file slug used on disk and in report names.
DATASETS: dict[str, str] = {
    "squad": "squad",
    "msmarcov1.1": "msmarcov1.1",
    "msmarcov2.1": "msmarcov2.1",
    "hotpotqa(distractor)": "hotpotqa_distractor",
    "race": "race",
    "duorc/selfrc": "duorc_selfrc",
    "duorc/paraphraserc": "duorc_paraphraserc",
    "musique": "musique",
    "2wikimultihopqa": "2wikimultihopqa",
}

#: The subset both shipped suites compare by default: the datasets with a full
#: validation split in ``1qa-clean-splits.manifest.json``.
DEFAULT_DATASETS: tuple[str, ...] = (
    "squad",
    "msmarcov1.1",
    "msmarcov2.1",
    "hotpotqa(distractor)",
    "race",
    "musique",
    "2wikimultihopqa",
)


def slug(dataset: str) -> str:
    """File/report name of one dataset, e.g. ``hotpotqa(distractor) -> hotpotqa_distractor``."""
    try:
        return DATASETS[dataset]
    except KeyError:
        raise ValueError(
            f"unknown dataset {dataset!r}; known datasets: {', '.join(sorted(DATASETS))}"
        ) from None


def split_file(root, dataset: str, split: str) -> Path:
    """The per-dataset raw file of one evaluation split (under ``meta/``)."""
    if split not in SPLIT_DIRS:
        raise ValueError(f"unknown evaluation split {split!r}; use test or validation")
    return Path(root) / "1qa" / "meta" / SPLIT_DIRS[split] / f"{slug(dataset)}.json"


def train_file(root) -> Path:
    """Filtered training split (``scripts/data/build_cache.py`` output)."""
    return Path(root) / "1qa" / "cache" / "train.jsonl"


def validation_file(root) -> Path:
    """Filtered validation split, i.e. the file an evaluation runs on."""
    return Path(root) / "1qa" / "cache" / "validation.jsonl"


def meta_file(root, split: str) -> Path:
    """Raw, unfiltered merged split (``meta_train``/``meta_validation``/``meta_test``)."""
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unknown split {split!r}; use train, validation or test")
    return Path(root) / "1qa" / "meta" / f"meta_{split}.json"


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def split_locations(split: str) -> tuple[tuple[str, str], ...]:
    """The canonical ``(directory, suffix)`` location for one evaluation split."""
    if split not in SPLIT_DIRS:
        raise ValueError(f"unknown evaluation split {split!r}; use test or validation")
    primary = ("meta/" + SPLIT_DIRS[split], ".json")
    return (primary,)


def resolve(root, dataset: str, split: str | None = None) -> tuple[Path, str]:
    """``(path, split)`` of one dataset's evaluation file.

    With ``split=None`` the fixed ``test`` split is preferred and ``validation`` is the
    fallback, which is the order the suites have always used when both exist.
    """
    candidates = (split,) if split else ("test", "validation")
    looked = []
    for candidate in candidates:
        for directory, suffix in split_locations(candidate):
            path = Path(root) / "1qa" / directory / f"{slug(dataset)}{suffix}"
            looked.append(path)
            if _nonempty(path):
                return path, candidate
    raise FileNotFoundError(
        f"no nonempty {split or 'test/validation'} file for dataset {dataset!r}; looked at "
        + ", ".join(str(path) for path in looked)
        + " -- build the corpus with `python scripts/data/build_1qa.py`"
    )


def report_name(dataset: str) -> str:
    """Compact JSON report filename of one dataset inside a checkpoint directory."""
    return f"{slug(dataset)}.json"
