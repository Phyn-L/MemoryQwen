from __future__ import annotations

from baselines.common import MemoryBaselineModel
from src.model import ContextPrefix


class Compressor500xModel(MemoryBaselineModel):
    """500x-style memory-position KV prefix with the shared training contract."""

    @classmethod
    def adapt(cls, model):
        model = super().adapt(model)
        model.__class__ = cls
        return model

    def encode_context_prefix(self, context_embeds, context_mask, **_kwargs):
        import torch
        from transformers.cache_utils import DynamicCache

        from src.model import compact_positions

        context_embeds = context_embeds.to(dtype=self.dtype)
        context_mask = context_mask.bool()
        batch, width, _ = context_embeds.shape
        memory = self.memory_embeddings.to(dtype=self.dtype).expand(batch, -1, -1)
        sequence = torch.cat((context_embeds, memory), dim=1)
        valid = torch.cat((context_mask, torch.ones(
            batch, self.memory_length, dtype=torch.bool, device=context_mask.device)), dim=1)
        with self.lora_enabled(True):
            out = self._transformer_body(
                inputs_embeds=sequence,
                attention_mask=None if bool(valid.all()) else valid,
                position_ids=compact_positions(valid, 0), use_cache=True, return_dict=True)
        cache = DynamicCache([
            (layer.keys[:, :, width:width + self.memory_length],
             layer.values[:, :, width:width + self.memory_length])
            for layer in out.past_key_values.layers
        ], config=self.qwen.config)
        states = out.last_hidden_state[:, width:, :]
        return ContextPrefix(states, width, self.memory_length,
                             memory_embedding=states.new_zeros(states.shape),
                             memory_hidden=states, past_key_values=cache)

    def _cached_decoder_pass(self, prefix, indices, sequence, sequence_mask, labels,
                             answer_length, loss_only, max_logits_rows,
                             return_stats=False):
        import torch

        from src.model import ContextMemoryOutput, compact_positions

        indices = indices.to(sequence.device, dtype=torch.long)
        cache = self._copy_memory_cache(prefix, indices)
        full_mask = torch.cat((sequence_mask.new_ones(
            sequence_mask.size(0), self.memory_length), sequence_mask), dim=1)
        positions = compact_positions(sequence_mask, self.memory_length)
        if int(positions.max()) >= self.window_limit:
            raise ValueError("memory cache + decoder sequence exceeds the backbone window")
        body = self._transformer_body if loss_only else self.qwen
        with self.lora_enabled(False):
            out = body(inputs_embeds=sequence, attention_mask=full_mask,
                       position_ids=positions, past_key_values=cache,
                       use_cache=False, return_dict=True)
        ignored = torch.full((labels.size(0), sequence.size(1) - answer_length), -100,
                             dtype=labels.dtype, device=labels.device)
        full_labels = torch.cat((ignored, labels), dim=1)
        losses = mask = stats = None
        if loss_only:
            from src.losses import sequence_lm_loss, teacher_forced_row_stats
            hidden, targets = out.last_hidden_state[:, :-1], full_labels[:, 1:]
            head = self.qwen.get_output_embeddings()
            if return_stats:
                stats = teacher_forced_row_stats(hidden, targets, head,
                                                 max_logits_rows=max_logits_rows)
                losses = stats["nll"] / stats["tokens"].clamp_min(1.0)
                mask = stats["tokens"] > 0
            else:
                losses, mask = sequence_lm_loss(hidden, targets, head,
                                                max_logits_rows=max_logits_rows)
        return ContextMemoryOutput(None if loss_only else out.logits, full_labels,
                                   prefix.memory, losses, mask, stats)

    def forward_with_memory_prefix(
        self, prefix, qa_context_indices, question_embeds, question_mask,
        answer_embeds, answer_mask, labels, loss_only=False,
        max_logits_rows=256, prompt_prefix_embeds=None, prompt_prefix_mask=None,
        answer_prefix_embeds=None, answer_prefix_mask=None,
        question_readout_mask=None, question_hidden=None,
    ):
        import torch

        del question_readout_mask, question_hidden
        question_embeds = question_embeds.to(dtype=self.dtype)
        answer_embeds = answer_embeds.to(dtype=self.dtype)
        question_mask = question_mask.bool()
        if prompt_prefix_embeds is None:
            prompt_prefix_embeds = question_embeds.new_empty(
                question_embeds.size(0), 0, self.qwen_hidden_size)
            prompt_prefix_mask = question_mask.new_empty(question_mask.size(0), 0)
        parts = [prompt_prefix_embeds.to(dtype=self.dtype), question_embeds]
        masks = [prompt_prefix_mask.bool(), question_mask]
        if answer_prefix_embeds is not None:
            parts.append(answer_prefix_embeds.to(dtype=self.dtype))
            masks.append(answer_prefix_mask.bool())
        parts.append(answer_embeds)
        masks.append(answer_mask.bool())
        sequence, sequence_mask = torch.cat(parts, 1), torch.cat(masks, 1)
        return self._cached_decoder_pass(prefix, qa_context_indices, sequence,
                                         sequence_mask, labels, answer_embeds.size(1),
                                         loss_only, max_logits_rows)

    def forward_recon_with_memory_prefix(
        self, prefix, recon_context_indices, context_embeds, context_mask,
        labels, loss_only=False, max_logits_rows=256, return_stats=False,
    ):
        import torch

        context_embeds = context_embeds.to(dtype=self.dtype)
        context_mask = context_mask.bool()
        task = self.ae_task_embedding.to(dtype=self.dtype).view(1, 1, -1).expand(context_embeds.size(0), -1, -1)
        embedding = self.qwen.get_input_embeddings()
        user = embedding(self.recon_user_token_ids.to(context_embeds.device))[None].expand(context_embeds.size(0), -1, -1)
        header = embedding(self.recon_answer_prefix_token_ids.to(context_embeds.device))[None].expand(context_embeds.size(0), -1, -1)
        sequence = torch.cat((user, task, header, context_embeds), dim=1)
        sequence_mask = torch.cat((context_mask.new_ones(
            context_mask.size(0), user.size(1) + 1 + header.size(1)), context_mask), dim=1)
        return self._cached_decoder_pass(prefix, recon_context_indices, sequence,
                                         sequence_mask, labels, context_embeds.size(1),
                                         loss_only, max_logits_rows, return_stats)

__all__ = ["Compressor500xModel"]
