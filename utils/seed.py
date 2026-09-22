"""Centralised random seeding for experiments."""

from __future__ import annotations

import os
import random

import torch


def seed_everything(seed: int, rank: int = 0, deterministic: bool = False) -> int:
    """Seed Python, NumPy (when installed), and PyTorch for one process.

    ``rank`` gives each distributed process an independent stream while keeping the
    run reproducible. Use the same seed across ranks for shared model initialisation.
    Call before CUDA work. Deterministic algorithm selection is opt-in because it
    can reduce performance and is not supported by every CUDA operation. False
    leaves existing backend settings unchanged. This does not restore checkpoint
    RNG states or seed independently constructed generators.
    """
    effective_seed = int(seed) + int(rank)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    random.seed(effective_seed)

    try:
        import numpy as np
    except ImportError:  # NumPy is optional for this project.
        pass
    else:
        np.random.seed(effective_seed % (2**32))

    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)

    return effective_seed
