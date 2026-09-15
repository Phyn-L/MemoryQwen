from __future__ import annotations

import torch
from torch import nn


class MemoryDecoder(nn.Module):
    """A single Qwen-layer memory decoder.

    Input is that layer's context-dependent memory embedding ``[B, M, H]``;
    learned positional queries produce the complete context embedding sequence
    ``[B, L, H]``.
    """

    def __init__(self, hidden_size: int, num_heads: int, max_context_tokens: int = 2048):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.position = nn.Embedding(max_context_tokens, hidden_size)
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size), nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

    def forward(self, memory_embedding: torch.Tensor, context_length: int) -> torch.Tensor:
        if memory_embedding.ndim != 3:
            raise ValueError("memory_embedding must have shape [B, M, H]")
        positions = torch.arange(context_length, device=memory_embedding.device)
        query = self.position(positions)[None].expand(memory_embedding.size(0), -1, -1)
        attended, _ = self.cross_attention(query, memory_embedding, memory_embedding, need_weights=False)
        hidden = self.norm(query + attended)
        return self.norm(hidden + self.ffn(hidden))


__all__ = ["MemoryDecoder"]
