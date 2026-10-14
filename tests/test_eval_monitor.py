from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.eval_monitor import EvalResourceMonitor


def test_monitor_writes_resource_and_liveness_snapshots(tmp_path):
    with EvalResourceMonitor(tmp_path, step=200, device="cpu", interval_seconds=60):
        pass

    log_path = tmp_path / "eval_resource_logs" / "step-200.jsonl"
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert rows
    rank = rows[-1]["ranks"][0]
    assert rows[-1]["step"] == 200
    assert rank["status"] == "alive"
    assert rank["process_alive"] is True
    assert rank["pid"] > 0
    assert rank["host"]
    assert rank["cpu_rss_bytes"] > 0
    assert rank["gpu"] is None


def test_monitor_marks_missing_rank_without_heartbeat(tmp_path):
    monitor = EvalResourceMonitor(tmp_path, step=400, device="cpu")
    monitor.world_size = 2

    assert monitor._rank_statuses()[1]["status"] == "missing"
    assert monitor._rank_statuses()[1]["process_alive"] is None
