from __future__ import annotations

import os


def init_distributed():
    """Initialize torch.distributed when launched with torchrun; no-op otherwise."""
    import torch.distributed as dist
    if not dist.is_available() or dist.is_initialized():
        return
    if "RANK" not in os.environ:
        return
    backend = "nccl" if __import__("torch").cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def is_main_process() -> bool:
    import torch.distributed as dist
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def barrier() -> None:
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
