"""Per-rank resource and liveness snapshots during autoregressive evaluation."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import threading
import time
from typing import Any

import torch
import torch.distributed as dist


def _rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _cpu_rss_bytes() -> int | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _resource_snapshot(device: torch.device) -> dict[str, Any]:
    gpu = None
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            gpu = {
                "device": str(device),
                "allocated_bytes": torch.cuda.memory_allocated(device),
                "reserved_bytes": torch.cuda.memory_reserved(device),
                "free_bytes": free_bytes,
                "total_bytes": total_bytes,
            }
        except (RuntimeError, AssertionError):
            gpu = {"device": str(device), "status": "unavailable"}
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "cpu_rss_bytes": _cpu_rss_bytes(),
        "gpu": gpu,
        "process_alive": True,
    }


class EvalResourceMonitor:
    """Write local rank heartbeats and aggregate rank liveness on rank zero."""

    def __init__(
        self,
        output_dir: str | Path,
        step: int,
        device: torch.device,
        interval_seconds: float = 10.0,
        stale_after_seconds: float = 30.0,
    ):
        self.rank, self.world_size = _rank_world()
        self.device = torch.device(device)
        self.interval_seconds = float(interval_seconds)
        self.stale_after_seconds = float(stale_after_seconds)
        self.log_dir = Path(output_dir) / "eval_resource_logs"
        self.heartbeat_dir = self.log_dir / f"step-{int(step)}.heartbeats"
        self.log_path = self.log_dir / f"step-{int(step)}.jsonl"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _heartbeat_path(self, rank: int) -> Path:
        return self.heartbeat_dir / f"rank-{rank:05d}.json"

    def _write_heartbeat(self) -> None:
        snapshot = _resource_snapshot(self.device)
        self.heartbeat_dir.mkdir(parents=True, exist_ok=True)
        path = self._heartbeat_path(self.rank)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(snapshot) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def _rank_statuses(self) -> list[dict[str, Any]]:
        now = time.time()
        statuses = []
        for rank in range(self.world_size):
            path = self._heartbeat_path(rank)
            try:
                snapshot = json.loads(path.read_text(encoding="utf-8"))
                age = max(0.0, now - path.stat().st_mtime)
                snapshot["heartbeat_age_seconds"] = round(age, 2)
                snapshot["status"] = (
                    "alive" if age <= self.stale_after_seconds else "stale"
                )
            except (OSError, ValueError):
                snapshot = {
                    "rank": rank,
                    "status": "missing",
                    "process_alive": None,
                    "heartbeat_age_seconds": None,
                }
            snapshot["rank"] = rank
            statuses.append(snapshot)
        return statuses

    def _append_aggregate(self) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": int(self.log_path.stem.removeprefix("step-")),
            "rank": self.rank,
            "world_size": self.world_size,
            "ranks": self._rank_statuses(),
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        states = ",".join(
            f"r{item['rank']}={item['status']}" for item in row["ranks"]
        )
        print(
            f"[eval-resource] step={row['step']} rank_status=[{states}] "
            f"log={self.log_path}",
            flush=True,
        )

    def _run(self) -> None:
        next_aggregate = 0.0
        while not self._stop.is_set():
            try:
                self._write_heartbeat()
                now = time.monotonic()
                if self.rank == 0 and now >= next_aggregate:
                    self._append_aggregate()
                    next_aggregate = now + self.interval_seconds
            except Exception as error:  # keep a diagnostic logger from stopping evaluation
                print(
                    f"[eval-resource] rank={self.rank} monitor_error="
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
            self._stop.wait(self.interval_seconds)

    def start(self) -> "EvalResourceMonitor":
        self._write_heartbeat()
        self._thread = threading.Thread(
            target=self._run,
            name=f"eval-resource-rank-{self.rank}",
            daemon=True,
        )
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
        try:
            self._write_heartbeat()
            if self.rank == 0:
                self._append_aggregate()
        except Exception as error:
            print(
                f"[eval-resource] rank={self.rank} final_snapshot_error="
                f"{type(error).__name__}: {error}",
                flush=True,
            )

    def __enter__(self) -> "EvalResourceMonitor":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.close()
