from __future__ import annotations

import torch
from torch import nn

from .dtypes import no_autocast


class MemoryDecoder(nn.Module):
    """A bottleneck decoder for one Qwen layer's memory states.

    Input is that layer's context-dependent memory embedding ``[B, M, H]``;
    memory and learned positional queries are projected to a smaller decoder
    width ``D`` before cross-attention and the FFN. The result is projected back
    to ``H`` to reconstruct the complete context embeddings ``[B, L, H]``.

    The decoder keeps its own parameter dtype, which may be float32 while the Qwen
    backbone is bfloat16. ``memory_embedding`` is therefore cast to the decoder's
    dtype on entry and the result is returned in that same dtype; the reconstruction
    loss upcasts both sides before reducing. The body runs with autocast disabled so
    that a mixed-precision training forward does not downcast it again.
    """

    def __init__(
        self,
        qwen_hidden_size: int,
        decoder_hidden_size: int,
        num_heads: int,
        ffn_ratio: int = 2,
        max_context_tokens: int = 2048,
        reconstruct_embeddings: bool = True,
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
        # Only the embedding-regression objective needs the projection back to the
        # backbone width. The context next-token objective classifies the bottleneck
        # hidden state directly through a shared vocabulary head, so building this
        # projection there would leave 28 * (D * H + H) parameters with no gradient
        # and, worse, a matching AdamW state allocation.
        self.output_projection = (
            nn.Linear(decoder_hidden_size, qwen_hidden_size) if reconstruct_embeddings else None
        )

    def decode(self, memory_embedding: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Decode arbitrary query positions into the bottleneck width ``[B, P, D]``.

        ``positions`` is a ``[B, P]`` (or ``[P]``) long tensor of context positions.
        The returned tensor is the pre-``output_projection`` hidden state, in the
        decoder's parameter dtype. The context next-token objective applies a shared
        vocabulary head to this tensor, so it must not be projected to ``H`` first.

        Queries attend to the memory only. That is deliberate: if the decoder could
        also attend to earlier context tokens it could act as a small standalone
        language model and the memory would receive almost no gradient.
        """
        if memory_embedding.ndim != 3:
            raise ValueError("memory_embedding must have shape [B, M, H]")
        positions = positions.long()
        if positions.ndim == 1:
            positions = positions[None].expand(memory_embedding.size(0), -1)
        with no_autocast(memory_embedding.device):
            dtype = self.memory_projection.weight.dtype
            memory = self.memory_projection(memory_embedding.to(dtype))
            query = self.position(positions)
            attended, _ = self.cross_attention(query, memory, memory, need_weights=False)
            hidden = self.norm(query + attended)
            return self.norm(hidden + self.ffn(hidden))

    def forward(self, memory_embedding: torch.Tensor, context_length: int) -> torch.Tensor:
        if memory_embedding.ndim != 3:
            raise ValueError("memory_embedding must have shape [B, M, H]")
        if self.output_projection is None:
            raise RuntimeError(
                "this decoder was built with reconstruct_embeddings=False; "
                "use decode() and the shared context_lm_head instead"
            )
        positions = torch.arange(context_length, device=memory_embedding.device)
        with no_autocast(memory_embedding.device):
            return self.output_projection(self.decode(memory_embedding, positions))


__all__ = ["MemoryDecoder"]
