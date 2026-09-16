"""Per-question read-out over the cached, question-agnostic memory (C3).

The global memory slots are produced without knowing the question, so they have to keep
everything that might be asked about; a single question typically needs a tiny fraction of a
long context. This module lets the question pick: ``R`` learnable latents cross-attend over
the concatenation of the question embeddings and the memory states, and the result is
inserted as ``R`` extra input positions between the question and the answer.

Two properties make this cheap:

* the memory stays question-agnostic and therefore cacheable -- only the ``R`` read-out
  positions are recomputed per question, from ``M + Q`` keys;
* the output lives in the backbone's *input-embedding* space, so the backbone computes the
  read-out positions' own keys and values. No per-layer prefix projection is needed, and the
  module adds ~4-8M trainable parameters instead of a second copy of the memory path.

The module keeps its own parameter dtype (float32 by default, like the decoders) and runs
with autocast disabled so a bf16 training forward cannot silently downcast it.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .dtypes import no_autocast


class _ResamplerBlock(nn.Module):
    """Pre-norm cross-attention followed by a pre-norm feed-forward block."""

    def __init__(self, hidden_size: int, num_heads: int, ffn_ratio: int = 2):
        super().__init__()
        ffn_hidden = hidden_size * ffn_ratio
        self.query_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden), nn.GELU(), nn.Linear(ffn_hidden, hidden_size),
        )

    def forward(self, latents, context, key_padding):
        query = self.query_norm(latents)
        keys = self.context_norm(context)
        attended, _ = self.cross_attention(
            query, keys, keys, key_padding_mask=key_padding, need_weights=False,
        )
        latents = latents + attended
        return latents + self.ffn(self.ffn_norm(latents))


class QuestionResampler(nn.Module):
    """Resample the memory (and the question) into a few question-conditioned positions.

    Both the question and the memory are projected down to ``width`` first (256 by default,
    the same bottleneck the per-layer decoders use), so the cross-attention blocks cost
    ~1M parameters instead of the ~34M a full-width block would need. The output projection
    is initialised at the *token-embedding* scale, because its result is inserted into the
    backbone's input stream: at step 0 the read-out is a small perturbation of the
    no-read-out path rather than a large out-of-distribution input.
    """

    def __init__(self, hidden_size: int, num_readout: int = 8, num_layers: int = 2,
                 num_heads: int = 4, width: int = 256, ffn_ratio: int = 2,
                 dtype: torch.dtype | None = None):
        super().__init__()
        if num_readout <= 0:
            raise ValueError("num_readout must be positive")
        if width <= 0:
            raise ValueError("width must be positive")
        if width % num_heads:
            raise ValueError("width must be divisible by num_heads")
        self.hidden_size = int(hidden_size)
        self.num_readout = int(num_readout)
        self.width = int(width)
        self.query_projection = nn.Linear(hidden_size, width)
        self.memory_projection = nn.Linear(hidden_size, width)
        self.latents = nn.Parameter(torch.randn(num_readout, width) * 0.02)
        self.input_norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList(
            [_ResamplerBlock(width, num_heads, ffn_ratio) for _ in range(max(1, int(num_layers)))]
        )
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, hidden_size)
        nn.init.normal_(self.output_projection.weight, std=0.02 / math.sqrt(width))
        nn.init.zeros_(self.output_projection.bias)
        if dtype is not None:
            self.to(dtype=dtype)

    def forward(self, question_embeds: torch.Tensor, question_mask: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Return ``[B, R, H]`` read-out embeddings in the module's dtype.

        ``memory`` is the question-agnostic memory representation ``[B, M, H]``; padded
        question positions are masked out of the attention. The memory is always present, so
        no row can end up with an all-masked key set.
        """
        if memory.ndim != 3:
            raise ValueError("memory must have shape [B, M, H]")
        if question_embeds.size(0) != memory.size(0):
            raise ValueError("question and memory must share the batch dimension")
        dtype = self.latents.dtype
        padding = torch.cat(
            [
                ~question_mask.bool(),
                torch.zeros(memory.size(0), memory.size(1), dtype=torch.bool, device=memory.device),
            ],
            dim=1,
        )
        latents = self.latents.unsqueeze(0).expand(memory.size(0), -1, -1)
        # Everything runs with autocast disabled -- including the input projections. Under
        # accelerate's bf16 autocast a Linear whose *weights* are float32 still returns
        # bfloat16, which then meets the float32 LayerNorms below and raises
        # "expected scalar type BFloat16 but found Float". The module owns its dtype.
        with no_autocast(memory.device):
            context = torch.cat(
                [
                    self.query_projection(question_embeds.to(dtype)),
                    self.memory_projection(memory.to(dtype)),
                ],
                dim=1,
            )
            latents = self.input_norm(latents)
            for block in self.blocks:
                latents = block(latents, context, padding)
            return self.output_projection(self.output_norm(latents))


__all__ = ["QuestionResampler"]
