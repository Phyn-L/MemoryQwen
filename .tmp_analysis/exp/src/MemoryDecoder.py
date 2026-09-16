from __future__ import annotations

import torch
from torch import nn


class MemoryDecoder(nn.Module):
    """A bottleneck decoder for one Qwen layer's memory states.

    Input is that layer's context-dependent memory embedding ``[B, M, H]``;
    memory and learned positional queries are projected to a smaller decoder
    width ``D`` before cross-attention and the FFN. The result is projected back
    to ``H`` to reconstruct the complete context embeddings ``[B, L, H]``.
    """

    def __init__(
        self,
        qwen_hidden_size: int,
        decoder_hidden_size: int,
        num_heads: int,
        ffn_ratio: int = 2,
        max_context_tokens: int = 2048,
    ):
        super().__init__()
        if decoder_hidden_size % num_heads:
            raise ValueError("decoder_hidden_size must be divisible by num_heads")
        if ffn_ratio <= 0:
            raise ValueError("ffn_ratio must be positive")
        ffn_hidden_size = decoder_hidden_size * ffn_ratio
        self.memory_projection = nn.Linear(qwen_hidden_size, decoder_hidden_size)
        self.position = nn.Embedding(max_context_tokens, decoder_hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            decoder_hidden_size, num_heads, batch_first=True,
        )
        self.norm = nn.LayerNorm(decoder_hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(decoder_hidden_size, ffn_hidden_size), nn.GELU(),
            nn.Linear(ffn_hidden_size, decoder_hidden_size),
        )
        self.output_projection = nn.Linear(decoder_hidden_size, qwen_hidden_size)

    def forward(self, memory_embedding: torch.Tensor, context_length: int) -> torch.Tensor:
        if memory_embedding.ndim != 3:
            raise ValueError("memory_embedding must have shape [B, M, H]")
        positions = torch.arange(context_length, device=memory_embedding.device)
        query = self.position(positions)[None].expand(memory_embedding.size(0), -1, -1)
        memory = self.memory_projection(memory_embedding)
        attended, _ = self.cross_attention(query, memory, memory, need_weights=False)
        hidden = self.norm(query + attended)
        hidden = self.norm(hidden + self.ffn(hidden))
        return self.output_projection(hidden)


__all__ = ["MemoryDecoder"]
