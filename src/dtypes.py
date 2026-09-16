"""Dtype helpers shared by the trainable adapter modules.

The backbone stays in its checkpoint dtype (bfloat16 for Qwen3) while the trainable
adapter parameters are kept in float32 so that AdamW's moments and its
``param.add_(update, alpha=-lr)`` step are not quantised to bfloat16 resolution. That
means the model has two dtypes at once, and every place where they meet needs an
explicit cast. :func:`no_autocast` is the second half of that contract: Accelerate
wraps the training forward in ``torch.autocast``, which would otherwise downcast the
float32 adapter matmuls back to bfloat16 and throw away the precision they exist for.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


@contextmanager
def no_autocast(device):
    """Run a block with autocast disabled, computing in the block's own dtype."""
    device_type = device if isinstance(device, str) else torch.device(device).type
    if device_type in {"cuda", "cpu"}:
        with torch.autocast(device_type=device_type, enabled=False):
            yield
    else:
        yield


__all__ = ["no_autocast"]
