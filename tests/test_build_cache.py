"""The filtered cache: raw meta splits in, ``cache/{train,validation}.jsonl`` out.

``src/cache_build.py`` is what turns the unfiltered ``meta/meta_<split>.json`` into the files a
run reads. Its job is to apply *exactly* the loader's filter (``data.filter_rows_batch``) and to
know when its output is stale; both are things a silent change would corrupt (a training run
would quietly see more or fewer rows than its config says).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.cache_build import (  # noqa: E402
    build,
    cache_fingerprint,
    cache_is_fresh,
    ensure_cache,
    filter_split,
)
from utils.config import DataConfig  # noqa: E402


class _Encoded:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class _Tokenizer:
    """Whitespace tokenizer: one token per word, enough for the length rules.

    Like a real tokenizer it returns flat ids for a single string and a list of them for a
    batch; getting that wrong made a length rule read ``len([[tokens]]) == 1``.
    """

    name_or_path = "cache-build-test"
    vocab_size = 32
    apply_chat_template = None

    def __call__(self, texts, **kwargs):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        limit = kwargs.get("max_length")
        encoded = [text.split()[:limit] if limit else text.split() for text in items]
        return _Encoded(encoded[0] if single else encoded)


def _row(context: str, questions, dataset: str = "squad", index: int = 0) -> dict:
    return {
        "id": f"{dataset}:train:{index}",
        "dataset": dataset,
        "source_split": "train",
        "context": context,
        "qa_pairs": [
            {
                "id": f"{dataset}:train:{index}:{position}",
                "question": question,
                "answers": ["a"],
                "answer": "a",
                "metadata": {},
            }
            for position, question in enumerate(questions)
        ],
        "metadata": {},
    }


def _meta(tmp_path: Path, split: str, rows) -> Path:
    path = tmp_path / "1qa" / "meta" / f"meta_{split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def _config(tmp_path: Path, **overrides) -> DataConfig:
    values = dict(
        root=str(tmp_path),
        max_context_tokens=8,
        max_question_tokens=8,
        max_answer_tokens=4,
        max_conversation_tokens=64,
        filter_long_context=True,
        filter_no_qa=True,
        train_datasets="all",
        validation_datasets=None,
        train_max_samples=None,
        validation_max_samples=None,
    )
    values.update(overrides)
    return DataConfig(**values)


def test_the_filter_keeps_only_rows_this_config_accepts(tmp_path):
    _meta(
        tmp_path,
        "train",
        [
            _row("short context", ["q1", "q2"], index=0),
            _row("one two three four five six seven eight nine ten eleven", ["q"], index=1),
            _row("   ", ["q"], index=2),  # blank context
            _row("no answers here", ["q"], index=3),
        ],
    )
    # `_row` gives every pair an answer, so the last row still passes; empty questions and
    # answers are already impossible in a meta file (the builder drops such pairs).
    config = _config(tmp_path)
    info = filter_split(config, "train", config.meta_file("train"), _Tokenizer(), quiet=True)
    assert info["contexts"] == 2
    assert info["qa_pairs"] == 3, "every QA pair of a kept context survives"
    rows = [json.loads(line) for line in Path(info["file"]).read_text(encoding="utf-8").splitlines()]
    assert [pair["question"] for pair in rows[0]["qa_pairs"]] == ["q1", "q2"]
    assert rows[0]["context"] == "short context"


def test_the_dataset_selection_of_each_split_is_honoured(tmp_path):
    _meta(tmp_path, "train", [_row("c one", ["q"], dataset="squad", index=0),
                              _row("c two", ["q"], dataset="race", index=1)])
    _meta(tmp_path, "validation", [_row("c three", ["q"], dataset="squad", index=2),
                                   _row("c four", ["q"], dataset="race", index=3)])
    config = _config(tmp_path, train_datasets=["race"], validation_datasets=["squad"])
    train = filter_split(config, "train", config.meta_file("train"), _Tokenizer(), quiet=True)
    validation = filter_split(
        config, "validation", config.meta_file("validation"), _Tokenizer(), quiet=True
    )
    assert train["contexts"] == 1 and validation["contexts"] == 1
    train_rows = Path(train["file"]).read_text(encoding="utf-8")
    validation_rows = Path(validation["file"]).read_text(encoding="utf-8")
    assert "c two" in train_rows and "c one" not in train_rows
    assert "c three" in validation_rows and "c four" not in validation_rows


def test_no_sample_count_is_baked_into_the_cache(tmp_path):
    """Sample caps are the loader's job; a capped cache could not be raised again."""
    _meta(tmp_path, "train", [_row(f"context {i}", ["q"], index=i) for i in range(5)])
    config = _config(tmp_path, train_max_samples=2)
    info = filter_split(config, "train", config.meta_file("train"), _Tokenizer(), quiet=True)
    assert info["contexts"] == 5


def test_a_missing_raw_split_says_how_to_build_it(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(FileNotFoundError, match="build_1qa.py"):
        filter_split(config, "validation", config.meta_file("validation"), _Tokenizer())


def test_freshness_tracks_the_config_and_the_raw_file(tmp_path):
    source = _meta(tmp_path, "train", [_row("short context", ["q"])])
    config = _config(tmp_path)
    tokenizer = _Tokenizer()
    assert not cache_is_fresh(config, "train", tokenizer)

    filter_split(config, "train", config.meta_file("train"), tokenizer, quiet=True)
    build(config, tokenizer, splits=["train"], quiet=True)
    assert cache_is_fresh(config, "train", tokenizer)

    # A different filter limit must invalidate the cache ...
    stricter = _config(tmp_path, max_context_tokens=2)
    assert not cache_is_fresh(stricter, "train", tokenizer)

    # ... and so must a changed raw file.
    added = json.dumps(_row("another", ["q"], index=9)) + "\n"
    source.write_text(source.read_text(encoding="utf-8") + added, encoding="utf-8")
    assert not cache_is_fresh(config, "train", tokenizer)


def test_ensure_cache_builds_once_and_then_reports_nothing_to_do(tmp_path):
    _meta(tmp_path, "train", [_row("short context", ["q"])])
    _meta(tmp_path, "validation", [_row("another context", ["q"], index=1)])
    config = _config(tmp_path)
    tokenizer = _Tokenizer()

    built = ensure_cache(config, tokenizer, quiet=True)
    assert sorted(built) == ["train", "validation"]
    assert Path(config.cache_file("train")).is_file()
    assert Path(config.cache_file("validation")).is_file()
    assert ensure_cache(config, tokenizer, quiet=True) == []

    # Removing one file rebuilds only that one.
    Path(config.cache_file("validation")).unlink()
    assert ensure_cache(config, tokenizer, quiet=True) == ["validation"]


def test_a_dry_run_writes_no_cache_file(tmp_path):
    _meta(tmp_path, "train", [_row("short context", ["q"])])
    config = _config(tmp_path)
    build(config, _Tokenizer(), splits=["train"], quiet=True, dry_run=True)
    assert not Path(config.cache_file("train")).exists()
    assert not Path(config.cache_file("train")).with_suffix(".jsonl.tmp").exists()


def test_the_fingerprint_ignores_unrelated_fields(tmp_path):
    source = _meta(tmp_path, "train", [_row("short context", ["q"])])
    tokenizer = _Tokenizer()
    config = _config(tmp_path)
    same = _config(
        tmp_path, num_workers=4, pin_memory=False, max_qa_sample_per_context=16,
    )
    assert cache_fingerprint(config, "train", source, tokenizer) == cache_fingerprint(
        same, "train", source, tokenizer
    )
    different = _config(tmp_path, max_answer_tokens=1)
    assert cache_fingerprint(config, "train", source, tokenizer) != cache_fingerprint(
        different, "train", source, tokenizer
    )


def test_parallel_filtering_is_byte_identical_to_serial(tmp_path):
    """A process pool must not reorder or change a single row of the cache."""
    rows = [
        _row(
            f"context {index} " + "word " * (index % 7),
            [f"q{index}", f"q{index}b"],
            index=index,
        )
        for index in range(60)
    ]
    serial_root, parallel_root = tmp_path / "serial", tmp_path / "parallel"
    for root in (serial_root, parallel_root):
        _meta(root, "train", rows)

    tokenizer = _Tokenizer()
    serial = filter_split(
        _config(serial_root), "train", _config(serial_root).meta_file("train"), tokenizer,
        quiet=True, write=True, num_workers=1, max_chunk_bytes=256,
    )
    parallel = filter_split(
        _config(parallel_root), "train", _config(parallel_root).meta_file("train"), tokenizer,
        quiet=True, write=True, num_workers=4, max_chunk_bytes=256,
    )
    assert serial["contexts"] == parallel["contexts"] > 0
    assert serial["rows_scanned"] == parallel["rows_scanned"]
    serial_bytes = Path(serial["file"]).read_bytes()
    parallel_bytes = Path(parallel["file"]).read_bytes()
    assert serial_bytes == parallel_bytes


def test_chunks_are_bounded_by_rows_and_bytes(tmp_path):
    """A pool task must not carry an unbounded amount of text (long passages are large)."""
    from src.cache_build import _iter_chunks

    path = tmp_path / "meta_train.json"
    path.write_text(
        "".join(json.dumps(_row("word " * 200, ["q"], index=i)) + "\n" for i in range(10)),
        encoding="utf-8",
    )
    chunks = list(_iter_chunks(path, None, max_rows=10, max_bytes=1024))
    assert len(chunks) > 1, "the byte cap must split before the row cap is reached"
    assert sum(len(chunk) for _, chunk in chunks) == 10
    assert [read for read, _ in chunks] == sorted(read for read, _ in chunks)
    assert chunks[-1][0] == 10, "the last chunk reports the rows read so far"


class _Bar:
    """Records updates from a progress renderer, without drawing anything."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.updates: list[int] = []
        self.postfix: list[dict] = []
        self.closed = False

    def update(self, amount):
        self.updates.append(amount)

    def set_postfix(self, **values):
        self.postfix.append(values)

    def close(self):
        self.closed = True


class _RichProgress:
    def __init__(self, bars, *args, **kwargs):
        self.bars = bars
        self.options = kwargs
        self.bar = None

    def add_task(self, desc, total, **fields):
        self.bar = _Bar(desc=desc, total=total, fields=fields)
        self.bars.append(self.bar)
        return 0

    def start(self):
        pass

    def update(self, task, advance=0, refresh=True, **fields):
        self.bar.updates.append(advance)
        if fields:
            self.bar.postfix.append(fields)

    def stop(self):
        self.bar.closed = True


def test_the_progress_bar_accounts_for_the_file_and_reports_the_counts(tmp_path, monkeypatch):
    import src.cache_build as cache_build

    made: list[_Bar] = []
    monkeypatch.setattr(cache_build, "tqdm", lambda **kwargs: made.append(_Bar(**kwargs)) or made[-1])
    source = _meta(tmp_path, "train", [
        _row("short context", ["q1", "q2"], index=0),
        _row("one two three four five six seven eight nine ten", ["q"], index=1),
    ])
    config = _config(tmp_path)
    info = filter_split(config, "train", config.meta_file("train"), _Tokenizer(), quiet=False)

    bar, = made
    assert bar.kwargs["total"] == source.stat().st_size, "the bar is the file size"
    assert bar.kwargs["desc"].endswith("meta_train.json")
    assert sum(bar.updates) == source.stat().st_size, "every byte of the file is accounted for"
    assert bar.postfix[-1] == {"rows": 2, "contexts": 1, "qa_pairs": 2}
    assert bar.closed
    assert info["contexts"] == 1


def test_quiet_and_no_progress_do_not_create_a_bar(tmp_path, monkeypatch):
    import src.cache_build as cache_build

    made: list[_Bar] = []
    monkeypatch.setattr(cache_build, "tqdm", lambda **kwargs: made.append(_Bar(**kwargs)) or made[-1])
    config = _config(tmp_path)
    _meta(tmp_path, "train", [_row("short context", ["q"])])

    filter_split(config, "train", config.meta_file("train"), _Tokenizer(), quiet=True)
    assert made == [], "--quiet must not draw"

    filter_split(
        config, "train", config.meta_file("train"), _Tokenizer(), quiet=False, progress=False
    )
    assert made == [], "--no-progress must not draw"


def test_the_loader_reads_a_fresh_cache_without_filtering_again(tmp_path):
    """A matching cache file must not be re-tokenized (that was minutes of duplicate work)."""
    from src.data import ConversationDataset
    from src.pipeline import uses_filtered_cache

    _meta(tmp_path, "train", [_row(f"context {i} " + "word " * i, [f"q{i}"], index=i)
                             for i in range(5)])
    config = _config(tmp_path)
    tokenizer = _Tokenizer()
    def full(data_config):
        # `uses_filtered_cache` takes the whole config, like `make_context_dataset` does.
        return SimpleNamespace(data=data_config)

    assert not uses_filtered_cache(full(config), "train", tokenizer), "no cache yet"

    build(config, tokenizer, splits=["train"], quiet=True)
    assert uses_filtered_cache(full(config), "train", tokenizer)

    cache_dir = tmp_path / "loader-cache"
    dataset = ConversationDataset(
        config.cache_file("train"), tokenizer=tokenizer, max_context_tokens=8,
        cache_dir=cache_dir, pre_filtered=True,
    )
    filtered = ConversationDataset(
        config.meta_file("train"), tokenizer=tokenizer, max_context_tokens=8,
        cache_dir=None,
    )
    assert [record.context for record in dataset] == [record.context for record in filtered]
    assert not list(cache_dir.glob("*")), "no second filtered copy is written"

    # The sample cap still applies to the direct read (and stops reading early).
    capped = ConversationDataset(
        config.cache_file("train"), tokenizer=tokenizer, max_context_tokens=8,
        cache_dir=None, max_samples=2, pre_filtered=True,
    )
    assert len(capped) == 2
    assert len(capped._rows) == 2, "a capped run must not read the whole file"


def test_a_stale_or_overridden_file_is_still_filtered(tmp_path):
    from src.pipeline import uses_filtered_cache

    def full(data_config):
        return SimpleNamespace(data=data_config)

    source = _meta(tmp_path, "train", [_row("short context", ["q"])])
    config = _config(tmp_path)
    tokenizer = _Tokenizer()
    build(config, tokenizer, splits=["train"], quiet=True)
    assert uses_filtered_cache(full(config), "train", tokenizer)

    # The raw meta file (what `--data-file` / `meta_test.json` point at) is not the cache.
    assert not uses_filtered_cache(full(config), "test", tokenizer)
    assert not uses_filtered_cache(
        full(_config(tmp_path, validation_datasets=["squad"])), "validation", tokenizer
    )

    # A changed filter limit makes the recorded fingerprint stale.
    assert not uses_filtered_cache(full(_config(tmp_path, max_context_tokens=4)), "train", tokenizer)

    # And so does a changed raw file.
    source.write_text(
        source.read_text(encoding="utf-8") + json.dumps(_row("added", ["q"], index=7)) + "\n",
        encoding="utf-8",
    )
    assert not uses_filtered_cache(full(config), "train", tokenizer)


def test_the_loader_preprocessing_cache_lives_with_the_data(tmp_path):
    """Loader-derived caches are not scattered into `outputs/`; they sit under the data root."""
    from src.pipeline import data_cache_dir

    config = _config(tmp_path)
    assert data_cache_dir(SimpleNamespace(data=config)) == tmp_path / "1qa" / "cache" / "loader"

    # An override wins (local scratch for a slow network data root).
    scratch = tmp_path / "scratch"
    overridden = _config(tmp_path, preprocess_cache_dir=str(scratch))
    assert data_cache_dir(SimpleNamespace(data=overridden)) == scratch


def test_a_filtered_row_cache_is_written_under_the_data_root(tmp_path):
    """The one case that still writes a JSONL: a file that needs the loader filter."""
    from src.pipeline import make_context_dataset

    meta = _meta(tmp_path, "train", [_row(f"context {i} " + "word " * i, [f"q{i}"], index=i)
                                    for i in range(4)])
    # A hand-pointed raw file (what `--data-file` does in the evaluation suites): it is not the
    # built cache, so the loader filters it and caches the result with the data.
    config = _config(tmp_path, cache_filtered_dataset=True, train_conversation_file=str(meta))
    dataset = make_context_dataset(SimpleNamespace(data=config), "train", _Tokenizer())
    assert len(dataset) == 4
    cache_dir = tmp_path / "1qa" / "cache" / "loader"
    written = list(cache_dir.glob("conversation_filtered-*.jsonl"))
    assert len(written) == 1, "the loader filter result is cached with the data"


def test_a_large_cache_read_draws_a_progress_bar(tmp_path, monkeypatch):
    """The up-front parse of the built cache is the step that used to be silent."""
    import src.data as data_module
    from src.data import ConversationDataset

    made: list[_Bar] = []
    monkeypatch.setattr(
        data_module, "Progress", lambda *args, **kwargs: _RichProgress(made, *args, **kwargs)
    )
    source = _meta(tmp_path, "train", [_row("short context", ["q1", "q2"], index=0)])
    monkeypatch.setattr(ConversationDataset, "READ_BAR_MIN_BYTES", 1)

    dataset = ConversationDataset(source, tokenizer=_Tokenizer(), cache_dir=None, pre_filtered=True)
    assert len(dataset) == 1
    bar, = made
    assert bar.kwargs["total"] == source.stat().st_size
    assert bar.kwargs["desc"] == f"Reading {source.name}"
    assert sum(bar.updates) == source.stat().st_size, "every byte is accounted for"
    assert bar.postfix[-1] == {"rows": "1 rows"}
    assert bar.closed


def test_a_capped_read_stays_silent_and_stops_early(tmp_path, monkeypatch):
    """`--max-samples` reads only a prefix, so it neither needs nor draws the bar."""
    import src.data as data_module
    from src.data import ConversationDataset

    made: list[_Bar] = []
    monkeypatch.setattr(
        data_module, "Progress", lambda *args, **kwargs: _RichProgress(made, *args, **kwargs)
    )
    source = _meta(tmp_path, "train", [_row(f"context {i}", [f"q{i}"], index=i) for i in range(5)])
    monkeypatch.setattr(ConversationDataset, "READ_BAR_MIN_BYTES", 1)

    dataset = ConversationDataset(
        source, tokenizer=_Tokenizer(), cache_dir=None, pre_filtered=True, max_samples=2
    )
    assert len(dataset) == 2
    assert made == [], "a capped read must not promise a whole-file bar"


def _context_dataset(cache_path, tokenizer, config):
    """A `ConversationDataset` over a built cache, i.e. what the sampler really receives."""
    from src.data import ConversationDataset

    return ConversationDataset(
        cache_path, tokenizer=tokenizer, max_context_tokens=config.max_context_tokens,
        cache_dir=None, pre_filtered=True,
    )


# --------------------------------------------------------------- context length sidecar
def test_the_filter_hands_back_the_lengths_it_measured(tmp_path):
    from src.data import filter_options, filter_rows_batch

    rows = [
        _row("short context", ["q"], index=0),
        _row("one two three four five six seven eight nine ten eleven", ["q"], index=1),
        _row("another short one", ["q"], index=2),
    ]
    options = filter_options(8, True, True, 64, 8, 4)
    lengths: list[int] = []
    accepted = filter_rows_batch(rows, _Tokenizer(), options, lengths)
    assert [row["id"] for row in accepted] == ["squad:train:0", "squad:train:2"]
    assert lengths == [2, 3]
    # The accepted rows all fit the cap, so the truncated count *is* the true length.
    from src.data import context_lengths

    assert context_lengths([row["context"] for row in accepted], _Tokenizer()) == lengths

    # With the length rule off there is nothing to record.
    unknown: list[int] = []
    filter_rows_batch(rows, _Tokenizer(), filter_options(8, False, True, 64, 8, 4), unknown)
    assert unknown == [-1, -1, -1]


def test_parallel_and_serial_lengths_agree(tmp_path):
    from src.data import filter_options, filter_row_chunks

    rows = [_row(f"context {i} " + "word " * i, ["q"], index=i) for i in range(30)]
    options = filter_options(64, True, True, 512, 64, 8)

    def run(num_workers):
        chunks = [rows[index:index + 7] for index in range(0, len(rows), 7)]
        rows_out, lengths = [], []
        for accepted, chunk_lengths in filter_row_chunks(
            iter(chunks), _Tokenizer(), options, num_workers, with_lengths=True
        ):
            rows_out.extend(accepted)
            lengths.extend(chunk_lengths)
        return [row["id"] for row in rows_out], lengths

    assert run(0) == run(4)


def test_the_cache_build_records_lengths_next_to_the_file(tmp_path):
    import json as _json

    from src.data import context_lengths, context_lengths_path, read_context_lengths

    _meta(tmp_path, "train", [_row(f"context {i} " + "word " * i, [f"q{i}"], index=i)
                             for i in range(6)])
    config = _config(tmp_path)
    tokenizer = _Tokenizer()
    build(config, tokenizer, splits=["train"], quiet=True)

    cache_path = Path(config.cache_file("train"))
    sidecar = context_lengths_path(cache_path)
    assert sidecar.is_file(), "build_cache must record the lengths it measured"

    payload = _json.loads(sidecar.read_text())
    assert payload["size"] == cache_path.stat().st_size
    assert payload["mtime_ns"] == cache_path.stat().st_mtime_ns
    assert payload["max_context_tokens"] == config.max_context_tokens
    cached_rows = [json.loads(line) for line in cache_path.read_text().splitlines()]
    assert payload["lengths"] == [len(row["context"].split()) for row in cached_rows]

    dataset = _context_dataset(cache_path, tokenizer, config)
    assert read_context_lengths(dataset, tokenizer) == payload["lengths"]
    # The recorded lengths are exactly what measuring the contexts would produce.
    assert payload["lengths"] == context_lengths(
        [row["context"] for row in cached_rows], tokenizer
    )


def test_a_stale_or_mismatched_sidecar_is_ignored(tmp_path):
    import json as _json

    from src.data import context_lengths_path, read_context_lengths

    _meta(tmp_path, "train", [_row("short context", ["q"], index=0)])
    config = _config(tmp_path)
    tokenizer = _Tokenizer()
    build(config, tokenizer, splits=["train"], quiet=True)
    cache_path = Path(config.cache_file("train"))
    sidecar = context_lengths_path(cache_path)
    dataset = _context_dataset(cache_path, tokenizer, config)
    assert read_context_lengths(dataset, tokenizer) is not None

    # The described file changed -> the lengths may not match its rows any more.
    original = _json.loads(sidecar.read_text())
    original_stat = cache_path.stat()
    cache_path.write_text(cache_path.read_text() + "\n", encoding="utf-8")
    assert read_context_lengths(dataset, tokenizer) is None

    cache_path.write_text(cache_path.read_text().rstrip("\n") + "\n", encoding="utf-8")
    import os

    os.utime(cache_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    sidecar.write_text(_json.dumps(original), encoding="utf-8")
    assert read_context_lengths(dataset, tokenizer) is not None

    # A different tokenizer, a different context limit, an old version, too few lengths.
    for mutate in (
        lambda payload: payload.update(tokenizer="different"),
        lambda payload: payload.update(max_context_tokens=999),
        lambda payload: payload.update(version=0),
        lambda payload: payload.update(lengths=payload["lengths"][:0]),
    ):
        payload = _json.loads(sidecar.read_text())
        mutate(payload)
        sidecar.write_text(_json.dumps(payload), encoding="utf-8")
        assert read_context_lengths(dataset, tokenizer) is None

    sidecar.write_text(_json.dumps(original), encoding="utf-8")
    assert read_context_lengths(dataset, tokenizer) == original["lengths"]


def test_the_sortish_sampler_reuses_the_recorded_lengths(tmp_path, monkeypatch):
    """The whole point: no `Preparing sortish lengths` pass when the build recorded them."""
    import src.data as data_module
    from src.data import SortishSampler, context_lengths_path

    _meta(tmp_path, "train", [_row(f"context {i} " + "word " * i, [f"q{i}"], index=i)
                             for i in range(5)])
    config = _config(tmp_path)
    tokenizer = _Tokenizer()
    build(config, tokenizer, splits=["train"], quiet=True)
    cache_path = Path(config.cache_file("train"))
    recorded = json.loads(context_lengths_path(cache_path).read_text())["lengths"]

    bars: list[_Bar] = []
    monkeypatch.setattr(
        data_module, "tqdm", lambda **kwargs: bars.append(_Bar(**kwargs)) or bars[-1]
    )
    dataset = _context_dataset(cache_path, tokenizer, config)
    sampler = SortishSampler(dataset, tokenizer, batch_size=1, cache_dir=None)
    assert sampler.lengths == recorded
    assert [bar.kwargs.get("desc") for bar in bars] == [], "no measuring pass"

    # Without the sidecar the sampler measures (and shows its bar) as before.
    context_lengths_path(cache_path).unlink()
    sampler = SortishSampler(dataset, tokenizer, batch_size=1, cache_dir=None)
    assert sampler.lengths == recorded
    assert [bar.kwargs.get("desc") for bar in bars] == ["Preparing sortish lengths"]


def test_lengths_can_be_backfilled_for_an_existing_cache(tmp_path):
    from src.cache_build import ensure_cache, write_lengths_for_cache
    from src.data import context_lengths_path

    _meta(tmp_path, "train", [_row(f"context {i} " + "word " * i, [f"q{i}"], index=i)
                             for i in range(4)])
    config = _config(tmp_path)
    tokenizer = _Tokenizer()
    build(config, tokenizer, splits=["train"], quiet=True)
    cache_path = Path(config.cache_file("train"))
    sidecar = context_lengths_path(cache_path)
    expected = json.loads(sidecar.read_text())["lengths"]
    sidecar.unlink()

    # A fresh cache must not be re-filtered just to get the sidecar back.
    before = cache_path.stat().st_mtime_ns
    assert ensure_cache(config, tokenizer, splits=["train"], quiet=True) == []
    assert cache_path.stat().st_mtime_ns == before, "the cache file itself was left alone"
    assert json.loads(sidecar.read_text())["lengths"] == expected

    # And the standalone backfill writes the same thing.
    sidecar.unlink()
    assert write_lengths_for_cache(config, tokenizer, "train", quiet=True) == sidecar
    assert json.loads(sidecar.read_text())["lengths"] == expected
