from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
import tempfile

from src.data_sampling import sample_validation_by_dataset


def test_parallel_and_serial_sampling_are_identical():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.jsonl"
        rows = [
            {"dataset": dataset, "id": f"{dataset}-{index}", "index": index}
            for dataset in ("squad", "race", "msmarcov1.1")
            for index in range(30)
        ]
        source.write_text("".join(json.dumps(row) + "\n" for row in rows))
        datasets = {"squad", "race", "msmarcov1.1"}
        limits = {dataset: 5 for dataset in datasets}

        serial_path, serial_counts = sample_validation_by_dataset(
            source, root / "serial", limits, seed=271, datasets=datasets, num_workers=1,
        )
        parallel_path, parallel_counts = sample_validation_by_dataset(
            source, root / "parallel", limits, seed=271, datasets=datasets, num_workers=3,
        )

        assert serial_counts == parallel_counts == limits
        assert serial_path.read_bytes() == parallel_path.read_bytes()
        assert len(multiprocessing.active_children()) == 0


def test_cache_reuse_does_not_rewrite_manifest_or_jsonl():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.jsonl"
        source.write_text("".join(
            json.dumps({"dataset": "squad", "id": str(index)}) + "\n"
            for index in range(10)
        ))
        output = root / "output"
        path, _ = sample_validation_by_dataset(
            source, output, {"squad": 2}, seed=11, datasets={"squad"}, num_workers=2,
        )
        files = [output / "manifest.json", path, output / "squad.jsonl"]
        before = {file: file.read_bytes() for file in files}
        sample_validation_by_dataset(
            source, output, {"squad": 2}, seed=11, datasets={"squad"}, num_workers=2,
        )
        assert {file: file.read_bytes() for file in files} == before
