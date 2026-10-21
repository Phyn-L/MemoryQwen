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

import torch
import torch.nn.functional as F
from torch import nn

from .dtypes import no_autocast


class SharedMemoryOutputProjection(nn.Module):
    """Project resampler latents into the backbone hidden space.

    The module operates on the final dimension, so it supports both the current
    ``[B, M, latent_size]`` output and a future layer-conditioned
    ``[B, L, M, latent_size]`` representation without changing its interface.
    """

    def __init__(self, latent_size: int, hidden_size: int, ffn_ratio: int = 4):
        super().__init__()
        if latent_size <= 0 or hidden_size <= 0 or ffn_ratio <= 0:
            raise ValueError("latent_size, hidden_size and ffn_ratio must be positive")
        ffn_size = latent_size * ffn_ratio
        self.norm = nn.LayerNorm(latent_size)
        self.fc1 = nn.Linear(latent_size, ffn_size)
        self.fc2 = nn.Linear(ffn_size, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = F.silu(self.fc1(x))
        x = self.fc2(x)
        return self.output_norm(x)


class _ResamplerBlock(nn.Module):
    """Pre-norm attention block, optionally with latent self-attention first."""

    def __init__(self, hidden_size: int, num_heads: int, ffn_ratio: int = 2,
                 use_self_attention: bool = False):
        super().__init__()
        ffn_hidden = hidden_size * ffn_ratio
        self.self_query_norm = nn.LayerNorm(hidden_size) if use_self_attention else None
        self.self_attention = (
            nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
            if use_self_attention else None
        )
        self.query_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden), nn.GELU(), nn.Linear(ffn_hidden, hidden_size),
        )

    def forward(self, latents, context, key_padding):
        if self.self_attention is not None:
            self_query = self.self_query_norm(latents)
            attended, _ = self.self_attention(
                self_query, self_query, self_query, need_weights=False,
            )
            latents = latents + attended
        query = self.query_norm(latents)
        keys = self.context_norm(context)
        attended, _ = self.cross_attention(
            query, keys, keys, key_padding_mask=key_padding, need_weights=False,
        )
        latents = latents + attended
        return latents + self.ffn(self.ffn_norm(latents))


class QuestionResampler(nn.Module):
    """Resample the memory (and the question) into a few question-conditioned positions.

    The frozen LLM encodes the concatenated memory and question; learnable latents
    cross-attend to those final-layer states. They are projected down to the configured
    bottleneck (256 by default), so the cross-attention blocks cost
    ~1M parameters instead of the ~34M a full-width block would need. The output projection
    uses the same MLP head as the context resampler.
    """

    def __init__(self, hidden_size: int, question_resampler_length: int = 8,
                 question_resampler_layers: int = 2, question_resampler_heads: int = 4,
                 question_resampler_hidden_size: int = 256,
                 question_resampler_ffn_ratio: int = 2,
                 question_resampler_use_self_attention: bool = True,
                 dtype: torch.dtype | None = None, mode: str = "memory_question"):
        super().__init__()
        if mode not in {"memory_question", "memory_only"}:
            raise ValueError(f"invalid resampler mode: {mode}")
        self.mode = mode
        if question_resampler_length <= 0:
            raise ValueError("question_resampler_length must be positive")
        if question_resampler_hidden_size <= 0:
            raise ValueError("question_resampler_hidden_size must be positive")
        if question_resampler_layers <= 0 or question_resampler_heads <= 0:
            raise ValueError("question_resampler_layers and question_resampler_heads must be positive")
        if question_resampler_ffn_ratio <= 0:
            raise ValueError("question_resampler_ffn_ratio must be positive")
        if question_resampler_hidden_size % question_resampler_heads:
            raise ValueError("question_resampler_hidden_size must be divisible by question_resampler_heads")
        self.hidden_size = int(hidden_size)
        self.question_resampler_length = int(question_resampler_length)
        self.question_resampler_hidden_size = int(question_resampler_hidden_size)
        self.memory_projection = nn.Linear(hidden_size, question_resampler_hidden_size)
        self.latents = nn.Parameter(torch.randn(question_resampler_length, question_resampler_hidden_size) * 0.02)
        self.input_norm = nn.LayerNorm(question_resampler_hidden_size)
        self.blocks = nn.ModuleList(
            [_ResamplerBlock(question_resampler_hidden_size, question_resampler_heads,
                             question_resampler_ffn_ratio,
                             use_self_attention=question_resampler_use_self_attention)
             for _ in range(question_resampler_layers)]
        )
        self.output_projection = SharedMemoryOutputProjection(
            question_resampler_hidden_size, hidden_size,
        )
        if dtype is not None:
            self.to(dtype=dtype)

    def forward(self, context_hidden: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        """Return ``[B, R, H]`` read-out embeddings in the module's dtype.

        ``context_hidden`` contains final-layer LLM states for memory and question tokens,
        with shape ``[B, M + Q, H]``. Padded question positions are masked out.
        """
        if context_hidden.ndim != 3:
            raise ValueError("context_hidden must have shape [B, M + Q, H]")
        if context_mask.shape != context_hidden.shape[:2]:
            raise ValueError("context_mask must have shape [B, M + Q]")
        dtype = self.latents.dtype
        padding = ~context_mask.bool()
        latents = self.latents.unsqueeze(0).expand(context_hidden.size(0), -1, -1)
        # Everything runs with autocast disabled -- including the input projections. Under
        # accelerate's bf16 autocast a Linear whose *weights* are float32 still returns
        # bfloat16, which then meets the float32 LayerNorms below and raises
        # "expected scalar type BFloat16 but found Float". The module owns its dtype.
        with no_autocast(context_hidden.device):
            context = self.memory_projection(context_hidden.to(dtype))
            latents = self.input_norm(latents)
            for block in self.blocks:
                latents = block(latents, context, padding)
            return self.output_projection(latents)


class ContextMemoryResampler(nn.Module):
    """Compress arbitrary in-window context states into a fixed memory sequence.

    The context is encoded once by the backbone.  This module is deliberately
    independent of the backbone's KV cache: learned queries cross-attend to all
    valid context hidden states and emit ``num_memory`` embeddings in the
    backbone input space.  Each configured block runs
    ``latent self-attention -> context cross-attention -> FFN``.  A short
    backbone pass then turns those embeddings into the reusable per-layer
    memory KV cache.
    """

    def __init__(self, hidden_size: int, num_memory: int, num_layers: int = 2,
                 num_heads: int = 4, width: int = 256,
                 ffn_ratio: int = 2, dtype: torch.dtype | None = None,
                 backbone_layers: int | None = None,
                 output_mode: str = "layer0",
                 use_self_attention: bool = True):
        super().__init__()
        if output_mode not in {"layer0", "layerwise"}:
            raise ValueError("output_mode must be 'layer0' or 'layerwise'")
        if num_memory <= 0:
            raise ValueError("num_memory must be positive")
        if num_heads <= 0 or num_layers <= 0 or ffn_ratio <= 0:
            raise ValueError("num_heads, num_layers and ffn_ratio must be positive")
        if width <= 0 or width % num_heads:
            raise ValueError("width must be positive and divisible by num_heads")
        self.hidden_size = int(hidden_size)
        self.num_memory = int(num_memory)
        self.num_layers = int(num_layers)
        self.width = int(width)
        self.backbone_layers = None if backbone_layers is None else int(backbone_layers)
        self.output_mode = output_mode
        self.latents = nn.Parameter(torch.randn(num_memory, width) * 0.02)
        self.context_projection = nn.Linear(hidden_size, width)
        self.input_norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList(
            [_ResamplerBlock(width, num_heads, ffn_ratio,
                             use_self_attention=use_self_attention)
             for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = SharedMemoryOutputProjection(width, hidden_size)
        self.layerwise_output_projection = (
            SharedMemoryOutputProjection(width, hidden_size)
            if output_mode == "layerwise" else None
        )
        self.layer_embedding = None
        self.layer_gates = None
        if output_mode == "layerwise":
            if self.backbone_layers is None:
                raise ValueError("layerwise output requires backbone_layers")
            if self.backbone_layers <= 0:
                raise ValueError("backbone_layers must be positive")
            self.layer_embedding = nn.Parameter(torch.zeros(self.backbone_layers, width))
            self.layer_gates = nn.Parameter(torch.zeros(self.backbone_layers))
            # The layerwise head replaces the shared layer-0 head. Keeping the
            # inactive head trainable makes DDP report unused parameters every step.
            for parameter in self.output_projection.parameters():
                parameter.requires_grad = False
        else:
            for parameter in (
                self.layerwise_output_projection.parameters()
                if self.layerwise_output_projection is not None
                else ()
            ):
                parameter.requires_grad = False
        if dtype is not None:
            self.to(dtype=dtype)

    def forward(self, context_hidden: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        if context_hidden.ndim != 3:
            raise ValueError("context_hidden must have shape [B, L, H]")
        if context_mask.shape != context_hidden.shape[:2]:
            raise ValueError("context_mask must have shape [B, L]")
        dtype = self.latents.dtype
        with no_autocast(context_hidden.device):
            context = self.context_projection(context_hidden.to(dtype))
            latents = self.input_norm(self.latents).unsqueeze(0).expand(context.size(0), -1, -1)
            padding = ~context_mask.bool()
            for block in self.blocks:
                latents = block(latents, context, padding)
            latents = self.output_norm(latents)
            if self.output_mode == "layer0":
                return self.output_projection(latents)
            conditioned = latents[:, None, :, :] + self.layer_embedding[None, :, None, :]
            return self.layerwise_output_projection(conditioned)


class LearnedPrefix(nn.Module):
    """Shared input embeddings; no question or memory content is read here."""

    def __init__(self, hidden_size, num_readout, dtype=None):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_readout, hidden_size, dtype=dtype) * 0.02)

    def forward(self, question_embeds, question_mask, memory):
        return self.latents.unsqueeze(0).expand(question_embeds.size(0), -1, -1)


__all__ = [
    "ContextMemoryResampler", "QuestionResampler", "LearnedPrefix",
    "SharedMemoryOutputProjection",
]
