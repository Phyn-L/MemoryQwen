"""Experiment run naming and checkpoint/logging directory configuration."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import torch

MODEL_LABEL_PATTERN = re.compile(
    r"Qwen(?:3)?[-_]?([0-9]+(?:\.[0-9]+)?)[Bb]", re.IGNORECASE
)


def model_run_label(model_path: str) -> str:
    """``.../models--Qwen--Qwen3-8B/snapshots/...`` -> ``Qwen8B``."""
    match = MODEL_LABEL_PATTERN.search(model_path)
    return f"Qwen{match.group(1)}B" if match else "Qwen"


def new_run_name(model_path: str) -> str:
    """``Qwen8B_20260916_213847`` — default experiment name and checkpoint directory."""
    return f"{model_run_label(model_path)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def configure_run_paths(cfg, accelerator, resume: str | None, run_name_factory=new_run_name) -> None:
    """Resolve checkpoint and experiment log directories for a training run."""
    if resume:
        run_dir = Path(resume).resolve().parent
        if cfg.data.train_stage == "two_stage" and run_dir.name in {"recon", "qa"}:
            run_dir = run_dir.parent
        cfg.checkpoint.output_dir = str(run_dir)
        cfg.logging.swanlab_run_name = cfg.logging.swanlab_run_name or run_dir.name
        cfg.logging.swanlab_log_dir = cfg.checkpoint.output_dir
        return
    is_main = accelerator is None or accelerator.is_main_process
    run_name = run_name_factory(cfg.model.name_or_path) if is_main else None
    num_processes = accelerator.num_processes if accelerator is not None else 1
    if num_processes > 1:
        names = [run_name]
        torch.distributed.broadcast_object_list(names, src=0)
        run_name = names[0]
    cfg.logging.swanlab_run_name = cfg.logging.swanlab_run_name or run_name
    cfg.checkpoint.output_dir = str(
        Path(cfg.checkpoint.output_dir)
        / model_run_label(cfg.model.name_or_path)
        / run_name
    )
    cfg.logging.swanlab_log_dir = cfg.checkpoint.output_dir


__all__ = ["configure_run_paths", "model_run_label", "new_run_name"]
