from __future__ import annotations

import torch

from src.model import (
    ContextMemoryModel,
    ContextMemoryOutput,
    build_causal_mask,
    compact_positions,
)


class MemoryBaselineModel(ContextMemoryModel):
    """Shared task-marker and reconstruction behavior for memory baselines."""

    @property
    def recon_task_embedding(self):
        # The shared loop reports this field; baseline training has one canonical AE marker.
        return self.ae_task_embedding

    @classmethod
    def adapt(cls, model):
        memory_length = int(model.context_resampler.num_memory)
        model.__class__ = cls
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.ae_task_embedding = model.recon_task_embedding
        del model.recon_task_embedding
        model.context_resampler = torch.nn.Identity()
        model.resampler = None
        model.memory_length = memory_length
        model.memory_embeddings = torch.nn.Parameter(torch.empty(
            1, memory_length, model.qwen_hidden_size,
            device=model.qwen.get_input_embeddings().weight.device,
            dtype=model.trainable_dtype,
        ))
        torch.nn.init.normal_(model.memory_embeddings, mean=0.0, std=1.0)
        for name, parameter in model.qwen.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                parameter.requires_grad_(True)
        model.memory_embeddings.requires_grad_(True)
        model.ae_task_embedding.requires_grad_(True)
        return model

    def _reconstruction_sequence(self, prefix, indices, context_embeds, context_mask):
        context_embeds = context_embeds.to(dtype=self.dtype)
        context_mask = context_mask.bool()
        indices = indices.to(context_embeds.device, dtype=torch.long)
        task = self.ae_task_embedding.to(dtype=self.dtype).view(1, 1, -1).expand(context_embeds.size(0), -1, -1)
        memory = prefix.memory_embedding.index_select(0, indices).to(dtype=self.dtype)
        embedding = self.qwen.get_input_embeddings()
        user = embedding(self.recon_user_token_ids.to(context_embeds.device))[None].expand(context_embeds.size(0), -1, -1)
        header = embedding(self.recon_answer_prefix_token_ids.to(context_embeds.device))[None].expand(context_embeds.size(0), -1, -1)
        sequence = torch.cat([memory, user, task, header, context_embeds], dim=1)
        valid = torch.cat([
            context_mask.new_ones(context_mask.size(0), memory.size(1) + user.size(1) + 1 + header.size(1)),
            context_mask,
        ], dim=1)
        return sequence, valid

    def forward_recon_with_memory_prefix(
        self, prefix, recon_context_indices, context_embeds, context_mask, labels,
        loss_only=False, max_logits_rows=256, return_stats=False,
    ):
        if not self.recon_user_token_ids.numel() or not self.recon_answer_prefix_token_ids.numel():
            raise RuntimeError("Recon requires chat-template headers")
        sequence, valid = self._reconstruction_sequence(
            prefix, recon_context_indices, context_embeds, context_mask
        )
        if int(compact_positions(valid, 0).max()) >= self.window_limit:
            raise ValueError("memory + recon task + context exceeds the backbone window")
        body = self._transformer_body if loss_only else self.qwen
        with self.lora_enabled(False):
            out = body(inputs_embeds=sequence,
                       attention_mask=build_causal_mask(valid, sequence.dtype),
                       position_ids=compact_positions(valid, 0), use_cache=False,
                       return_dict=True)
        context_embeds = context_embeds.to(dtype=self.dtype)
        ignored = torch.full((labels.size(0), sequence.size(1) - context_embeds.size(1)),
                             -100, dtype=labels.dtype, device=labels.device)
        full_labels = torch.cat([ignored, labels], dim=1)
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
