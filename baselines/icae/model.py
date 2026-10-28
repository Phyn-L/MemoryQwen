from __future__ import annotations

import torch

from baselines.common import MemoryBaselineModel
from src.model import ContextPrefix, compact_positions


class IcaeModel(MemoryBaselineModel):
    """ICAE-style hidden-state memory writer over the project training contract."""

    @classmethod
    def adapt(cls, model):
        return super().adapt(model)

    def encode_context_prefix(self, context_embeds, context_mask, **_kwargs):
        context_embeds = context_embeds.to(dtype=self.dtype)
        context_mask = context_mask.bool()
        batch, width, _ = context_embeds.shape
        memory = self.memory_embeddings.to(dtype=self.dtype).expand(batch, -1, -1)
        sequence = torch.cat((context_embeds, memory), dim=1)
        valid = torch.cat(
            (context_mask, torch.ones(batch, self.memory_length, dtype=torch.bool,
                                      device=context_mask.device)), dim=1
        )
        with self.lora_enabled(True):
            hidden = self._transformer_body(
                inputs_embeds=sequence,
                attention_mask=None if bool(valid.all()) else valid,
                position_ids=compact_positions(valid, 0),
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
        states = hidden[:, width:, :]
        return ContextPrefix(states, width, self.memory_length,
                             memory_embedding=states)

