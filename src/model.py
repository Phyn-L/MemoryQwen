from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

from .dtypes import no_autocast
from .resampler import ContextMemoryResampler, LearnedPrefix, QuestionResampler


@dataclass
class ContextMemoryOutput:
    logits: torch.Tensor | None
    labels: torch.LongTensor
    memory: torch.Tensor
    supervision_losses: torch.Tensor | None = None
    # Which supervision rows carried a target (see losses.supervision_objectives).
    supervision_mask: torch.Tensor | None = None
    # Per-row (nll, correct, tokens) of the teacher-forced reconstruction, when requested.
    row_stats: dict | None = None



@dataclass
class ContextPrefix:
    memory: torch.Tensor
    context_length: int
    memory_length: int
    memory_embedding: torch.Tensor | None = None
    memory_hidden: torch.Tensor | None = None
    question_hidden: torch.Tensor | None = None
    past_key_values: object = None


@dataclass
class ConversationState:
    memory: torch.Tensor
    memory_hidden: torch.Tensor
    past_key_values: object
    history_ids: torch.LongTensor
    turn_count: int = 0


def compact_positions(
    valid: torch.Tensor, start: int, pad_position: int | None = None
) -> torch.Tensor:
    """Absolute position of every slot of one block that begins at ``start``.

    Real slots are numbered contiguously from ``start`` (the rank of the slot among the valid
    ones), padding slots get ``pad_position`` (default ``start``) and are always masked out of
    attention, so they only need to be *defined*, never meaningful.

    This is the single place the model decides where a token sits. Context encoding, the QA
    block, the read-out, the Recon task marker and the answer all use it, so the numbering a
    block gets cannot differ between training, encoding and generation -- and because the
    numbering is rank-based it is identical whether the padding of a block sits on the left or
    on the right.
    """
    valid = valid.bool()
    if valid.ndim != 2:
        raise ValueError("compact_positions expects a [B, S] mask")
    rank = valid.long().cumsum(dim=1) - 1
    pad = start if pad_position is None else int(pad_position)
    return torch.where(valid, start + rank, torch.full_like(rank, pad))


TASK_RECON = 0
TASK_QA = 1

def is_trainable_parameter_name(name: str) -> bool:
    """Single source of truth for which parameters are trained.

    Used both by :meth:`ContextMemoryModel.set_trainable_dtype` and by ``load_model`` so that a
    newly added trainable module cannot be forgotten in one of the two places.
    """
    return (
        "lora_A" in name
        or "lora_B" in name
        or name.startswith(("resampler.",))
        or name.startswith(("context_resampler.", "context_layer_logits"))
        or name == "recon_task_embedding"
    )


class MetaLoRA(nn.Module):
    """A regular hand-written LoRA layer for context adaptation.

    The frozen base layer keeps the backbone dtype, while ``lora_A``/``lora_B`` may be
    float32 so that their AdamW state is not quantised to bfloat16. The two dtypes are
    reconciled at the module boundary: the LoRA branch computes in its own dtype with
    autocast disabled and the delta is cast back before it is added to the base output.
    The adapter follows the usual LoRA computation while keeping the trainable branch in its configured dtype.
    """
    def __init__(
        self, base: nn.Linear, rank: int, alpha: float,
        dropout: float = 0.0, dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.enabled = True
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features, dtype=dtype))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank, dtype=dtype))
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        for p in base.parameters():
            p.requires_grad = False
    def forward(self, x):
        base_dtype = self.base.weight.dtype
        base_out = self.base(x.to(base_dtype))
        if not self.enabled:
            return base_out
        with no_autocast(x.device):
            lora_x = self.dropout(x).to(self.lora_A.dtype)
            delta = self.scaling * (lora_x @ self.lora_A.t() @ self.lora_B.t())
        return base_out + delta.to(base_dtype)


def build_causal_mask(valid: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Causal mask for a complete embedding sequence with padded rows."""
    valid = valid.bool()
    bsz, length = valid.shape
    causal = torch.tril(torch.ones((length, length), dtype=torch.bool, device=valid.device))
    allowed = valid[:, None, :] & causal[None]
    self_edge = torch.eye(length, dtype=torch.bool, device=valid.device)[None]
    allowed = torch.where(valid[:, :, None], allowed, self_edge)
    mask = torch.zeros((bsz, 1, length, length), dtype=dtype, device=valid.device)
    return mask.masked_fill(~allowed[:, None], torch.finfo(dtype).min)


class ContextMemoryModel(nn.Module):
    """Qwen encoder/decoder with hand-written MetaLoRA.

    Two dtypes coexist. The frozen Qwen backbone stays in its checkpoint dtype
    (``qwen_dtype``, bfloat16 for Qwen3) and every tensor that enters it is cast to
    that dtype. The trainable pieces -- learned memory queries, task embeddings and
    ``lora_A``/``lora_B`` -- can be kept in ``trainable_dtype`` (float32 by
    default) so that AdamW's moments and its ``param.add_(update, alpha=-lr)`` step
    are not quantised to bfloat16 resolution. The casts that reconcile the two dtypes
    live in the context resampler and complete sequence passes,
    ``MetaLoRA`` casts on entry and exit, and the losses
    upcast before reducing.
    """
    def __init__(
        self, qwen: nn.Module, rank=8, alpha=16.0, memory_length=8,
        target_modules=None, dropout=0.0, max_context_tokens=None,
        trainable_dtype=torch.float32,
        task_init_token_ids: tuple[int, ...] | None = None,
        readout_length=0, question_resampler_layers=2, question_resampler_heads=4,
        question_resampler_hidden_size=256, question_resampler_ffn_ratio=2,
        question_resampler_use_self_attention=True,
        readout_keep_question_tokens=None,
        readout_mode="memory_question", context_layer_indices=(-1,),
        memory_resampler_layers=2, memory_resampler_heads=4,
        memory_resampler_hidden_size=256, memory_resampler_ffn_ratio=2,
        memory_resampler_use_self_attention=True,
        question_resampler_info_source=None, context_gradient_checkpointing=False,
        gradient_checkpoint_every_n_layers=1,
        conversation_enabled=False, recompress_threshold=2048,
        max_recompress_tokens=2048,
        recon_user_token_ids: tuple[int, ...] = (),
        recon_answer_prefix_token_ids: tuple[int, ...] = (),
    ):
        super().__init__()
        self.rank, self.alpha = rank, alpha
        self.target_modules = tuple(
            ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
            if target_modules is None
            else target_modules
        )
        self.trainable_dtype = trainable_dtype
        # Which LoRA implementation is active, recorded so a run is reproducible.
        self.lora_backend = "manual" if self.target_modules else "none"
        self.qwen = self._add_lora(qwen, rank, alpha, dropout)
        self.register_buffer(
            "recon_user_token_ids",
            torch.tensor(recon_user_token_ids, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "recon_answer_prefix_token_ids",
            torch.tensor(recon_answer_prefix_token_ids, dtype=torch.long),
            persistent=False,
        )
        # `_transformer_body` resolves and caches the headless decoder stack on first use; it
        # is not assigned here because that would register it as a duplicate submodule.
        object.__setattr__(self, "_body", None)
        qwen_dtype = self.qwen.get_input_embeddings().weight.dtype
        self.qwen_hidden_size = self.qwen.config.hidden_size
        # ``window_limit`` is the backbone's positional window (nothing may exceed it);
        # ``max_context_tokens`` is the *training* cap on a context, which the data pipeline
        # shares. They are stored separately so the constructor argument cannot be silently
        # dropped: a context longer than the configured cap is an error, while a prompt plus
        # memory plus target is only bounded by the window.
        self.window_limit = int(self.qwen.config.max_position_embeddings)
        if max_context_tokens is not None and not 0 < max_context_tokens <= self.window_limit:
            raise ValueError(f"max_context_tokens must be within backbone window {self.window_limit}")
        self.max_context_tokens = (
            self.window_limit if max_context_tokens is None else int(max_context_tokens)
        )
        self.conversation_enabled = bool(conversation_enabled)
        self.recompress_threshold = int(recompress_threshold)
        self.max_recompress_tokens = int(max_recompress_tokens)
        if self.conversation_enabled and (
            self.recompress_threshold <= 0 or self.max_recompress_tokens < memory_length
        ):
            raise ValueError("conversation token limits must include the memory slots")
        if self.conversation_enabled and (
            self.recompress_threshold > self.window_limit
            or self.max_recompress_tokens > self.window_limit
        ):
            raise ValueError("conversation token limits must fit the backbone window")
        if not 0 < memory_length <= self.window_limit:
            raise ValueError("memory_length must be within the backbone window")
        self.context_gradient_checkpointing = bool(context_gradient_checkpointing)
        if type(gradient_checkpoint_every_n_layers) is not int or gradient_checkpoint_every_n_layers < 0:
            raise ValueError(
                "gradient_checkpoint_every_n_layers must be a non-negative integer"
            )
        self.gradient_checkpoint_every_n_layers = gradient_checkpoint_every_n_layers
        self.context_layer_indices = tuple(context_layer_indices)
        depth = int(self.qwen.config.num_hidden_layers)
        if not self.context_layer_indices or any(
            type(i) is not int or not -depth <= i < depth for i in self.context_layer_indices
        ):
            raise ValueError(f"context_layer_indices must contain integers in [-{depth}, {depth - 1}]")
        if len({i % depth for i in self.context_layer_indices}) != len(self.context_layer_indices):
            raise ValueError("context_layer_indices must select distinct layers")
        self.context_layer_logits = nn.Parameter(torch.zeros(len(self.context_layer_indices), dtype=trainable_dtype))
        self.context_resampler = ContextMemoryResampler(
            self.qwen_hidden_size, memory_length, memory_resampler_layers,
            memory_resampler_heads, memory_resampler_hidden_size,
            memory_resampler_ffn_ratio, dtype=trainable_dtype,
            use_self_attention=memory_resampler_use_self_attention,
        )
        # Recon's user-side task marker starts from the mean embedding of its instruction phrase.
        # ``load_model`` resolves those ids with the run's own tokenizer.
        embedding = self.qwen.get_input_embeddings().weight.detach()
        self.task_init_token_ids: tuple[int, ...] = ()
        task_init = None
        ids = tuple(int(value) for value in (task_init_token_ids or ()))
        if ids and min(ids) >= 0 and max(ids) < embedding.size(0):
            task_init = embedding[list(ids)].mean(dim=0)
            self.task_init_token_ids = ids
        if task_init is None:
            task_init = torch.empty(
                self.qwen_hidden_size, device=embedding.device, dtype=embedding.dtype
            )
            nn.init.normal_(task_init, mean=0.0, std=0.02)
        self.recon_task_embedding = nn.Parameter(task_init.to(dtype=trainable_dtype).clone())
        # Qwen's embedding weight is the single source of truth for the frozen
        # backbone dtype. It must not drag the trainable modules back to bfloat16,
        # so the trainable dtype is restored immediately afterwards.
        #
        # Per-question read-out over the question-agnostic memory. Zero means
        # the historical behaviour: the answer attends only to the memory and the question.
        if readout_mode not in {"memory_question", "learned_prefix", "off"}:
            raise ValueError(f"invalid readout_mode: {readout_mode}")
        if readout_length < 0:
            raise ValueError("readout_length must be non-negative")
        self.readout_mode = readout_mode
        if readout_mode == "memory_question":
            if question_resampler_info_source is None:
                question_resampler_info_source = "hidden"
            readout_keep_question_tokens = (
                True if readout_keep_question_tokens is None else readout_keep_question_tokens
            )
            if question_resampler_info_source not in {"hidden", "embedding"}:
                raise ValueError("question_resampler_info_source must be hidden or embedding")
        else:
            question_resampler_info_source = None
            readout_keep_question_tokens = True
        self.readout_keep_question_tokens = bool(readout_keep_question_tokens)
        self.question_resampler_info_source = question_resampler_info_source
        self.readout_length = 0 if readout_mode == "off" else int(readout_length)
        self.resampler = None
        if self.readout_length > 0:
            if readout_mode == "learned_prefix":
                self.resampler = LearnedPrefix(self.qwen_hidden_size, self.readout_length, dtype=trainable_dtype)
            else:
                self.resampler = QuestionResampler(
                    self.qwen_hidden_size,
                    question_resampler_length=self.readout_length,
                    question_resampler_layers=question_resampler_layers,
                    question_resampler_heads=question_resampler_heads,
                    question_resampler_hidden_size=question_resampler_hidden_size,
                    question_resampler_ffn_ratio=question_resampler_ffn_ratio,
                    question_resampler_use_self_attention=question_resampler_use_self_attention,
                    dtype=trainable_dtype,
                )
        self.to(dtype=qwen_dtype)
        self.set_trainable_dtype(trainable_dtype)

    def set_trainable_dtype(self, dtype: torch.dtype):
        """Cast every trainable parameter to ``dtype`` (float32 by default).

        The backbone is untouched. Called automatically at the end of ``__init__`` so
        that the global ``self.to(dtype=qwen_dtype)`` cast does not flatten the
        trainable parameters back to the backbone dtype. It matches the hand-written MetaLoRA parameters by name.
        """
        self.trainable_dtype = dtype
        for name, parameter in self.named_parameters():
            if is_trainable_parameter_name(name):
                parameter.data = parameter.data.to(dtype)
        return self

    def trainable_parameter_dtypes(self) -> dict[str, str]:
        dtypes: dict[str, str] = {}
        for _name, parameter in self.named_parameters():
            if parameter.requires_grad:
                dtypes[str(parameter.dtype)] = dtypes.get(str(parameter.dtype), 0) + 1
        return dtypes

    def _add_lora(self, qwen, rank, alpha, dropout):
        """Attach the hand-written MetaLoRA layers to the frozen backbone."""
        if not self.target_modules:
            for p in qwen.parameters():
                p.requires_grad = False
            return qwen
        for p in qwen.parameters():
            p.requires_grad = False
        for name, module in list(qwen.named_modules()):
            if not isinstance(module, nn.Linear) or not any(
                name.endswith(x) for x in self.target_modules
            ):
                continue
            parent=qwen
            for part in name.split('.')[:-1]:
                parent = getattr(parent, part)
            setattr(parent, name.split('.')[-1], MetaLoRA(module, rank, alpha, dropout, dtype=self.trainable_dtype))
        return qwen

    @contextmanager
    def lora_enabled(self, enabled: bool):
        """Temporarily enable or disable every MetaLoRA adapter.

        The context encoder and hidden readout inputs use LoRA-on. Decoder passes
        and decoder memory KV use LoRA-off to match the original frozen Qwen.
        """
        adapters = [module for module in self.qwen.modules() if isinstance(module, MetaLoRA)]
        previous = [module.enabled for module in adapters]
        try:
            for module in adapters:
                module.enabled = bool(enabled)
            yield self
        finally:
            for module, state in zip(adapters, previous, strict=True):
                module.enabled = state

    @property
    def dtype(self) -> torch.dtype:
        return self.qwen.get_input_embeddings().weight.dtype

    @property
    def _transformer_body(self):
        """The decoder stack without the language-model head, resolved once and retained.

        ``encode_context_prefix`` only reads ``hidden_states`` and ``past_key_values``, but
        ``load_model`` builds a ``Qwen3ForCausalLM``: calling ``self.qwen`` would additionally
        project every context position through the ``hidden x 151936`` vocabulary head (a
        wasted matmul plus a ~0.6 GiB logits tensor per row at context length 2048). See
        :meth:`_resolve_transformer_body` for how the wrapper is peeled and why an unknown
        wrapper is an error instead of a silent fallback.

        The reference is cached through ``object.__setattr__`` on purpose: assigning a module
        to a normal attribute would register it as a second submodule, so it would show up
        again in ``state_dict()``/``named_modules()`` even though it is the same object.
        """
        body = self.__dict__.get("_body")
        if body is None:
            body = self._resolve_transformer_body()
            object.__setattr__(self, "_body", body)
        return body

    def _resolve_transformer_body(self):
        """``self.qwen`` with its LM head removed, or a loud failure.

        Peeling the head off is a structural assumption about the transformers wrapper, so it
        is resolved once here and checked. The previous version walked the same wrappers on
        every call and silently returned ``self.qwen`` when it could not find them -- which
        turns "this transformers version wraps the model differently" into an
        out-of-memory error inside the first training step instead of a clear message.
        """
        module = self.qwen
        for _ in range(4):
            if hasattr(module, "layers"):
                return module
            inner = getattr(module, "base_model", None) or getattr(module, "model", None)
            if inner is None:
                break
            module = inner
        import transformers

        raise RuntimeError(
            f"cannot locate the transformer body of {type(self.qwen).__name__}: expected a "
            f"module exposing `.layers` (transformers {transformers.__version__}). Extend "
            "ContextMemoryModel._resolve_transformer_body for this wrapper."
        )

    def _encode_context_hidden(self, context_embeds, context_mask, positions):
        """Run the long context once while retaining only configured layer states."""
        body = self._transformer_body
        layers = getattr(body, "layers", None)
        if layers is None:
            raise RuntimeError("Qwen transformer body does not expose decoder layers")
        depth = len(layers)
        resolved = [index if index >= 0 else depth + index for index in self.context_layer_indices]
        if any(index < 0 or index >= depth for index in resolved):
            raise ValueError(
                f"context layer indices {self.context_layer_indices} are outside "
                f"[-{depth}, {depth - 1}]"
            )
        captured: dict[int, torch.Tensor] = {}
        hooks = []
        checkpointing = (
            self.context_gradient_checkpointing
            and self.gradient_checkpoint_every_n_layers > 0
            and self.training
            and torch.is_grad_enabled()
        )
        for index in sorted(set(resolved)):
            if index == depth - 1:
                continue
            def capture(_module, _inputs, output, layer_index=index):
                value = output[0] if isinstance(output, (tuple, list)) else output
                captured[layer_index] = value
            hooks.append(layers[index].register_forward_hook(capture))
        try:
            if checkpointing:
                body.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={
                        "use_reentrant": False,
                        "preserve_rng_state": True,
                    },
                    every_n_layers=self.gradient_checkpoint_every_n_layers,
                )
            out = body(
                inputs_embeds=context_embeds,
                attention_mask=None if bool(context_mask.all()) else context_mask,
                position_ids=positions,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
        finally:
            if checkpointing:
                body.gradient_checkpointing_disable()
                # Manual LoRA is not detected by Transformers' PEFT hook cleanup.
                body.disable_input_require_grads()
            for hook in hooks:
                hook.remove()
        captured[depth - 1] = out.last_hidden_state
        selected = []
        for index in resolved:
            if index not in captured:
                raise RuntimeError(f"Qwen did not produce requested context layer {index}")
            hidden = captured[index]
            if index != depth - 1:
                hidden = body.norm(hidden)
            selected.append(hidden)
        weights = self.context_layer_logits.softmax(dim=0)
        mixed = sum(
            weight.to(hidden.dtype) * hidden
            for weight, hidden in zip(weights, selected, strict=True)
        )
        return mixed, out.last_hidden_state

    def encode_context_prefix(
        self, context_embeds, context_mask, max_tokens=None,
        question_embeds=None, question_mask=None,
    ) -> ContextPrefix:
        """Encode contexts and optional independent questions in one padded batch."""
        if context_embeds.ndim != 3 or context_embeds.size(-1) != self.qwen_hidden_size:
            raise ValueError("context_embeds must have shape [B, L, backbone_hidden_size]")
        if context_mask.shape != context_embeds.shape[:2]:
            raise ValueError("context_mask must have shape [B, L]")
        context_limit = self.max_context_tokens if max_tokens is None else min(
            int(max_tokens), self.window_limit
        )
        if not 0 < context_embeds.size(1) <= context_limit:
            raise ValueError(
                f"context length {context_embeds.size(1)} must be within max_context_tokens "
                f"({context_limit}); the backbone window is "
                f"{self.window_limit}"
            )
        if not bool(context_mask.bool().any(dim=1).all()):
            raise ValueError("every context must contain at least one valid token")
        context_embeds = context_embeds.to(dtype=self.dtype)
        context_mask = context_mask.bool()
        # One standard causal Qwen pass over the context. The context KV is not cached.
        # Rank-based positions make the pass independent of the padding side: a right-padded
        # and a left-padded batch number the same real tokens the same way.
        context_count, context_width = context_mask.shape
        if question_embeds is not None:
            if question_embeds.ndim != 3 or question_embeds.size(0) == 0:
                raise ValueError("question_embeds must have shape [N, Q, H] with N > 0")
            if question_embeds.size(-1) != self.qwen_hidden_size:
                raise ValueError("question_embeds hidden size must match the backbone")
            if question_mask is None or question_mask.shape != question_embeds.shape[:2]:
                raise ValueError("question_mask must match question_embeds")
            question_embeds = question_embeds.to(device=context_embeds.device, dtype=self.dtype)
            question_mask = question_mask.bool().to(context_embeds.device)
        if question_embeds is not None:
            width = max(context_width, question_embeds.size(1))
            context_batch = torch.nn.functional.pad(
                context_embeds, (0, 0, 0, width - context_width)
            )
            context_valid = torch.nn.functional.pad(
                context_mask, (0, width - context_width)
            )
            question_batch = torch.nn.functional.pad(
                question_embeds, (0, 0, 0, width - question_embeds.size(1))
            )
            question_valid = torch.nn.functional.pad(
                question_mask, (0, width - question_mask.size(1))
            )
            encoder_embeds = torch.cat([context_batch, question_batch], dim=0)
            encoder_mask = torch.cat([context_valid, question_valid], dim=0)
        else:
            encoder_embeds, encoder_mask = context_embeds, context_mask
        positions = compact_positions(encoder_mask, 0)
        with self.lora_enabled(True):
            mixed_hidden, final_hidden = self._encode_context_hidden(
                encoder_embeds, encoder_mask, positions,
            )
        context_hidden = mixed_hidden[:context_count, :context_width]
        memory_embedding = self.context_resampler(context_hidden, context_mask).to(dtype=self.dtype)
        question_hidden = None
        memory_hidden = None
        memory_cache = None
        cache_memory = not torch.is_grad_enabled()
        hidden_readout = (
            self.resampler is not None and self.readout_mode == "memory_question"
            and self.question_resampler_info_source == "hidden"
        )
        if hidden_readout or cache_memory:
            if question_embeds is not None and self.question_resampler_info_source == "hidden":
                question_hidden = final_hidden[context_count:, :question_embeds.size(1)]
            memory_mask = torch.ones(
                memory_embedding.shape[:2], dtype=torch.bool, device=memory_embedding.device
            )
            memory_positions = compact_positions(memory_mask, 0)
            if hidden_readout:
                with self.lora_enabled(True):
                    memory_hidden = self._transformer_body(
                        inputs_embeds=memory_embedding,
                        position_ids=memory_positions,
                        use_cache=False, return_dict=True,
                    ).last_hidden_state
            if cache_memory:
                with self.lora_enabled(False):
                    memory_cache = self._transformer_body(
                        inputs_embeds=memory_embedding,
                        position_ids=memory_positions,
                        use_cache=True, return_dict=True,
                    ).past_key_values
        return ContextPrefix(
            memory_embedding,
            context_embeds.size(1), memory_embedding.size(1),
            memory_embedding=memory_embedding,
            memory_hidden=memory_hidden,
            question_hidden=question_hidden,
            past_key_values=memory_cache,
        )

    def _encode_independent_question_hidden(self, question_embeds, question_mask):
        """Encode only question tokens, independently from memory and chat markers."""
        if question_embeds.ndim != 3 or question_mask.shape != question_embeds.shape[:2]:
            raise ValueError("question embeddings and mask must have shapes [N, Q, H] and [N, Q]")
        positions = compact_positions(question_mask, 0)
        with self.lora_enabled(True):
            return self._transformer_body(
                inputs_embeds=question_embeds.to(dtype=self.dtype),
                attention_mask=question_mask.bool(),
                position_ids=positions,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state

    def _readout_embeddings(
        self, prefix, question_embeds, question_mask, indices=None,
        question_hidden=None,
    ):
        """Resample LLM hidden states for memory and the current question.

        ``indices`` selects which memory rows the questions belong to (the training collate
        flattens several QA rows per context, and generation groups rows by context), so the
        resampled memory always matches the question row it is paired with.
        """
        if self.resampler is None:
            return None
        if self.readout_mode == "learned_prefix":
            return self.resampler(question_embeds, question_mask, None).to(dtype=self.dtype)
        if prefix.memory is None:
            raise RuntimeError("the question read-out needs the memory representation")
        memory = (
            prefix.memory_embedding if self.question_resampler_info_source == "embedding"
            else prefix.memory_hidden
        )
        if memory is None:
            raise RuntimeError("the question read-out needs encoded memory hidden states")
        memory = memory if indices is None else memory.index_select(0, indices)
        memory_mask = torch.ones(memory.shape[:2], dtype=torch.bool, device=memory.device)
        question_mask = question_mask.bool()
        question_embeds = question_embeds.to(dtype=self.dtype)
        if self.question_resampler_info_source == "embedding":
            inputs = torch.cat([memory, question_embeds], dim=1)
            readout_mask = torch.cat([memory_mask, question_mask], dim=1)
        else:
            if question_hidden is None:
                question_hidden = self._encode_independent_question_hidden(
                    question_embeds, question_mask
                )
            inputs = torch.cat([memory, question_hidden], dim=1)
            readout_mask = torch.cat([memory_mask, question_mask], dim=1)
        readout = self.resampler(inputs, readout_mask)
        return readout.to(dtype=self.dtype)

    def forward_with_memory_prefix(
        self, prefix, qa_context_indices, question_embeds, question_mask,
        answer_embeds, answer_mask, labels, loss_only=False, max_logits_rows=256,
        prompt_prefix_embeds=None, prompt_prefix_mask=None,
        answer_prefix_embeds=None, answer_prefix_mask=None,
        question_readout_mask=None, question_hidden=None,
    ):
        question_embeds, answer_embeds = question_embeds.to(dtype=self.dtype), answer_embeds.to(dtype=self.dtype)
        indices = qa_context_indices.to(question_embeds.device, dtype=torch.long)
        question_mask = question_mask.bool()
        if prompt_prefix_embeds is None:
            prompt_prefix_embeds = question_embeds.new_empty((question_embeds.size(0), 0, self.qwen_hidden_size))
            prompt_prefix_mask = question_mask.new_empty((question_mask.size(0), 0))
        prompt_prefix_embeds = prompt_prefix_embeds.to(dtype=self.dtype)
        prompt_prefix_mask = prompt_prefix_mask.bool()
        readout = self._readout_embeddings(
            prefix, question_embeds,
            question_mask if question_readout_mask is None else question_readout_mask, indices,
            question_hidden=question_hidden,
        )
        keep_question = readout is None or self.readout_keep_question_tokens
        memory = prefix.memory_embedding.index_select(0, indices).to(dtype=self.dtype)
        parts = [memory, prompt_prefix_embeds]
        masks = [torch.ones(memory.shape[:2], dtype=torch.bool, device=memory.device), prompt_prefix_mask]
        if keep_question:
            parts.append(question_embeds)
            masks.append(question_mask)
        if readout is not None:
            parts.append(readout)
            masks.append(torch.ones(readout.shape[:2], dtype=torch.bool, device=readout.device))
        if answer_prefix_embeds is not None:
            parts.append(answer_prefix_embeds.to(dtype=self.dtype))
            masks.append(answer_prefix_mask.bool())
        parts.append(answer_embeds)
        masks.append(answer_mask)
        sequence = torch.cat(parts, dim=1)
        full_mask = torch.cat(masks, dim=1)
        mask = build_causal_mask(full_mask, sequence.dtype)
        positions = compact_positions(full_mask, 0).to(sequence.device)
        if int(positions.max()) >= self.window_limit:
            raise ValueError("memory + prompt + readout + target exceeds the backbone window")
        body = self._transformer_body if loss_only else self.qwen
        with self.lora_enabled(False):
            out = body(
                inputs_embeds=sequence, attention_mask=mask, position_ids=positions,
                use_cache=False, return_dict=True,
            )
        ignored = torch.full(
            (labels.size(0), sequence.size(1) - answer_embeds.size(1)), -100,
            dtype=labels.dtype, device=labels.device,
        )
        full_labels = torch.cat([ignored, labels], dim=1)
        row_losses = row_mask = None
        if loss_only:
            from .losses import sequence_lm_loss
            # One flattened projection for the whole batch; within it every row is averaged
            # over its own active tokens (then the caller averages rows per task), so a long
            # Recon target cannot outweigh a short QA answer.
            head = self.qwen.get_output_embeddings()
            row_losses, row_mask = sequence_lm_loss(
                out.last_hidden_state[:, :-1], full_labels[:, 1:], head,
                max_logits_rows=max_logits_rows,
            )
        return ContextMemoryOutput(
            logits=None if loss_only else out.logits,
            labels=full_labels,
            memory=prefix.memory,
            supervision_losses=row_losses,
            supervision_mask=row_mask,
        )

    def forward_recon_with_memory_prefix(
        self, prefix, recon_context_indices, context_embeds, context_mask,
        labels, loss_only=False, max_logits_rows=256, return_stats=False,
    ):
        """Reconstruct context in the assistant turn after a memory/task user turn."""
        if self.recon_user_token_ids.numel() == 0 or self.recon_answer_prefix_token_ids.numel() == 0:
            raise RuntimeError("Recon requires user and assistant headers from the chat template")
        context_embeds = context_embeds.to(dtype=self.dtype)
        context_mask = context_mask.bool()
        indices = recon_context_indices.to(context_embeds.device, dtype=torch.long)
        task = self.recon_task_embedding.to(dtype=self.dtype).view(1, 1, -1).expand(
            context_embeds.size(0), -1, -1
        )
        user_mask = torch.ones(
            (context_embeds.size(0), 1), dtype=torch.bool, device=context_embeds.device
        )
        memory = prefix.memory_embedding.index_select(0, indices).to(dtype=self.dtype)
        embedding = self.qwen.get_input_embeddings()
        user = embedding(self.recon_user_token_ids.to(context_embeds.device))[None].expand(
            context_embeds.size(0), -1, -1
        )
        header = embedding(
            self.recon_answer_prefix_token_ids.to(context_embeds.device)
        )[None].expand(context_embeds.size(0), -1, -1)
        sequence = torch.cat([memory, user, task, header, context_embeds], dim=1)
        full_mask = torch.cat([
            context_mask.new_ones(context_mask.size(0), memory.size(1) + user.size(1)),
            user_mask, context_mask.new_ones(context_mask.size(0), header.size(1)), context_mask,
        ], dim=1)
        mask = build_causal_mask(full_mask, sequence.dtype)
        positions = compact_positions(full_mask, 0)
        if int(positions.max()) >= self.window_limit:
            raise ValueError("memory + recon task + context exceeds the backbone window")
        body = self._transformer_body if loss_only else self.qwen
        with self.lora_enabled(False):
            out = body(
                inputs_embeds=sequence,
                attention_mask=mask,
                position_ids=positions,
                use_cache=False,
                return_dict=True,
            )
        ignored = torch.full(
            (labels.size(0), sequence.size(1) - context_embeds.size(1)), -100,
            dtype=labels.dtype, device=labels.device,
        )
        full_labels = torch.cat([ignored, labels], dim=1)
        row_losses = row_mask = row_stats = None
        if loss_only:
            from .losses import sequence_lm_loss, teacher_forced_row_stats
            head = self.qwen.get_output_embeddings()
            hidden, target = out.last_hidden_state[:, :-1], full_labels[:, 1:]
            if return_stats:
                row_stats = teacher_forced_row_stats(
                    hidden, target, head, max_logits_rows=max_logits_rows,
                )
                row_losses = row_stats["nll"] / row_stats["tokens"].clamp_min(1.0)
                row_mask = row_stats["tokens"] > 0
            else:
                row_losses, row_mask = sequence_lm_loss(
                    hidden, target, head, max_logits_rows=max_logits_rows,
                )
        return ContextMemoryOutput(
            logits=None if loss_only else out.logits,
            labels=full_labels,
            memory=prefix.memory,
            supervision_losses=row_losses,
            supervision_mask=row_mask,
            row_stats=row_stats,
        )

    @torch.no_grad()
    def recon_teacher_forced(self, prefix, context_embeds, context_mask, labels, max_logits_rows=256):
        """Memory-conditioned reconstruction score for the evaluation loop.

        One forward over ``[memory, recon_task_embedding, context]`` with the context tokens as
        the teacher-forced targets; returns the per-row ``{"nll", "correct", "tokens"}`` so the
        caller can aggregate whatever accuracy / perplexity it reports, per dataset. Nothing is
        decoded, which is what makes a Recon metric affordable inside the training loop.
        """
        indices = torch.arange(context_embeds.size(0), device=context_embeds.device)
        output = self.forward_recon_with_memory_prefix(
            prefix, indices, context_embeds, context_mask, labels,
            loss_only=True, max_logits_rows=max_logits_rows, return_stats=True,
        )
        return output.row_stats

    def forward(
        self, context_embeds, context_mask, question_embeds, question_mask,
        answer_embeds, answer_mask, labels, qa_context_indices=None,
        loss_config=None, loss_only=False,
        recon_batch=None, question_prefix_embeds=None,
        question_prefix_mask=None, answer_prefix_embeds=None, answer_prefix_mask=None,
        question_readout_mask=None,
    ):
        """Training path for QA and teacher-forced reconstruction."""
        encode_questions = (
            question_embeds.size(0) > 0 and self.resampler is not None
            and self.readout_mode == "memory_question"
            and self.question_resampler_info_source == "hidden"
        )
        prefix = self.encode_context_prefix(
            context_embeds, context_mask,
            question_embeds=question_embeds if encode_questions else None,
            question_mask=(question_mask if question_readout_mask is None else question_readout_mask)
            if encode_questions else None,
        )
        max_logits_rows = getattr(loss_config, "max_logits_rows", 256)
        indices = (
            qa_context_indices
            if qa_context_indices is not None
            else torch.arange(question_embeds.size(0), device=question_embeds.device)
        )
        if question_embeds.size(0):
            output = self.forward_with_memory_prefix(
                prefix, indices, question_embeds, question_mask, answer_embeds,
                answer_mask, labels, loss_only=loss_only, max_logits_rows=max_logits_rows,
                prompt_prefix_embeds=question_prefix_embeds,
                prompt_prefix_mask=question_prefix_mask,
                answer_prefix_embeds=answer_prefix_embeds,
                answer_prefix_mask=answer_prefix_mask,
                question_readout_mask=question_readout_mask,
                question_hidden=prefix.question_hidden,
            )
        else:
            output = ContextMemoryOutput(
                logits=None,
                labels=labels,
                memory=prefix.memory,
                supervision_losses=labels.new_empty((0,), dtype=torch.float32),
                supervision_mask=torch.zeros((0,), dtype=torch.bool, device=labels.device),
            )
        if recon_batch is not None and recon_batch["context_embeds"].size(0):
            recon_output = self.forward_recon_with_memory_prefix(
                prefix,
                recon_batch["context_indices"],
                recon_batch["context_embeds"],
                recon_batch["context_mask"],
                recon_batch["labels"],
                loss_only=loss_only,
                max_logits_rows=max_logits_rows,
            )
            if loss_only:
                output.supervision_losses = torch.cat(
                    [output.supervision_losses, recon_output.supervision_losses]
                )
                output.supervision_mask = torch.cat(
                    [output.supervision_mask, recon_output.supervision_mask]
                )
            else:
                output.logits = [output.logits, recon_output.logits]
                output.labels = [output.labels, recon_output.labels]
        return output

    @torch.no_grad()
    def generate_answers_with_prefix(self, prefix, qa_context_indices, question_ids, question_mask,
                                     tokenizer, max_new_tokens=128, group_by_context=True,
                                     max_rows_per_group=4, task_id=TASK_QA,
                                     question_prefix_ids=None, question_prefix_mask=None,
                                     answer_prefix_ids=None, answer_prefix_mask=None,
                                     question_readout_mask=None):
        with self.lora_enabled(False):
            return self._generate_answers_with_prefix(
                prefix, qa_context_indices, question_ids, question_mask,
                tokenizer, max_new_tokens, group_by_context, max_rows_per_group,
                task_id, question_prefix_ids, question_prefix_mask,
                answer_prefix_ids, answer_prefix_mask, question_readout_mask,
            )

    def _generate_answers_with_prefix(self, prefix, qa_context_indices, question_ids, question_mask,
                                      tokenizer, max_new_tokens=128, group_by_context=True,
                                      max_rows_per_group=4, task_id=TASK_QA,
                                      question_prefix_ids=None, question_prefix_mask=None,
                                      answer_prefix_ids=None, answer_prefix_mask=None,
                                      question_readout_mask=None):
        """Greedy decoding for many QA rows that share one context prefix.

        Rows belonging to the same context share memory embeddings, so they are
        prefilled and decoded as one batch instead of one row at a time. Each row keeps
        its own position ids, so a row's real question tokens occupy exactly the same
        absolute positions they would have in the row-by-row path; question padding is
        left of them and masked out. The batched result is therefore identical to
        decoding every row on its own, only faster -- ``tests/test_generation.py`` asserts
        that equality against ``group_by_context=False``.

        Only greedy decoding is implemented (``argmax`` per step, stop at EOS); sampling,
        beam search and length penalties are deliberately absent because every number this
        project reports is greedy. ``group_by_context=False`` decodes every row as its own group.
        """
        embedding = self.qwen.get_input_embeddings()
        device = question_ids.device
        eos_id = getattr(tokenizer, "eos_token_id", None)
        n_rows = question_ids.size(0)
        if question_prefix_ids is None:
            question_prefix_ids = question_ids.new_empty((n_rows, 0))
            question_prefix_mask = question_mask.new_empty((n_rows, 0))
        if task_id == TASK_RECON:
            if self.recon_user_token_ids.numel() == 0 or self.recon_answer_prefix_token_ids.numel() == 0:
                raise RuntimeError("Recon requires user and assistant headers from the chat template")
            question_prefix_ids = self.recon_user_token_ids.to(device)[None].expand(n_rows, -1)
            question_prefix_mask = torch.ones_like(question_prefix_ids, dtype=torch.bool)
            answer_prefix_ids = self.recon_answer_prefix_token_ids.to(device)[None].expand(n_rows, -1)
            answer_prefix_mask = torch.ones_like(answer_prefix_ids, dtype=torch.bool)

        if group_by_context:
            grouped: dict[int, list[int]] = {}
            for row in range(n_rows):
                grouped.setdefault(int(qa_context_indices[row]), []).append(row)
            groups = [grouped[key] for key in sorted(grouped)]
        else:
            groups = [[row] for row in range(n_rows)]
        max_rows_per_group = max(1, int(max_rows_per_group))

        # One padded [n_rows, max_new_tokens] buffer for the whole call, filled chunk by
        # chunk; `all_lengths` records how many tokens each row actually produced.
        max_new_tokens = max(0, int(max_new_tokens))
        result = torch.full(
            (n_rows, max_new_tokens), int(tokenizer.pad_token_id),
            dtype=torch.long, device=device,
        )
        all_lengths = torch.zeros(n_rows, dtype=torch.long, device=device)
        for group in groups:
            for chunk_start in range(0, len(group), max_rows_per_group):
                rows = group[chunk_start:chunk_start + max_rows_per_group]
                count = len(rows)
                row_index = torch.tensor(rows, dtype=torch.long, device=device)
                row_ids = question_ids[row_index]
                row_valid = question_mask[row_index].bool()
                prefix_ids_row = question_prefix_ids[row_index]
                prefix_valid_row = question_prefix_mask[row_index].bool()
                valid = row_valid.sum(dim=1)                              # [count]
                # Recon rows have no question tokens; keep the user segment exactly
                # ``[memory, recon_task_embedding]`` instead of adding a masked pad.
                width = int(valid.max())
                # Left-align the real question tokens; compact_positions then restores each
                # row's own absolute positions, so the layout matches the row-by-row decode.
                q_ids = torch.full((count, width), int(tokenizer.pad_token_id), dtype=row_ids.dtype, device=device)
                q_mask = torch.zeros((count, width), dtype=torch.bool, device=device)
                readout_q_mask = torch.zeros_like(q_mask)
                for slot in range(count):
                    real = row_ids[slot][row_valid[slot]]
                    q_ids[slot, width - real.numel():] = real
                    q_mask[slot, width - real.numel():] = True
                    readout_q_mask[slot, width - real.numel():] = (
                        True if question_readout_mask is None
                        else question_readout_mask[row_index[slot]][row_valid[slot]]
                    )
                question = embedding(q_ids).to(dtype=self.dtype)
                prefix_width = int(prefix_valid_row.sum(dim=1).max())
                prefix_ids = torch.full(
                    (count, prefix_width), int(tokenizer.pad_token_id),
                    dtype=prefix_ids_row.dtype, device=device,
                )
                prefix_mask = torch.zeros((count, prefix_width), dtype=torch.bool, device=device)
                for slot in range(count):
                    real = prefix_ids_row[slot][prefix_valid_row[slot]]
                    prefix_ids[slot, prefix_width - real.numel():] = real
                    prefix_mask[slot, prefix_width - real.numel():] = True
                prefix_embeds = embedding(prefix_ids).to(dtype=self.dtype)
                if task_id == TASK_RECON:
                    task = self.recon_task_embedding.to(dtype=self.dtype).view(1, 1, -1).expand(
                        count, -1, -1
                    )
                    question = torch.cat([task, question], dim=1)
                    task_mask = torch.ones((count, 1), dtype=torch.bool, device=device)
                    q_mask = torch.cat([task_mask, q_mask], dim=1)

                readout = None if task_id == TASK_RECON else self._readout_embeddings(
                    prefix, question, readout_q_mask,
                    qa_context_indices[row_index].to(device, dtype=torch.long),
                )
                # One helper numbers the whole prefill block: the read-out slots land right
                # after each row's real question tokens, exactly where a row-by-row decode
                # would put them, and the first generated token is then at ``start + real``.
                keep_question = readout is None or self.readout_keep_question_tokens
                memory = prefix.memory_embedding.index_select(
                    0, qa_context_indices[row_index].to(device, dtype=torch.long)
                ).to(dtype=self.dtype)
                sequence_parts = [memory, prefix_embeds]
                mask_parts = [torch.ones(memory.shape[:2], dtype=torch.bool, device=device), prefix_mask]
                if keep_question:
                    sequence_parts.append(question)
                    mask_parts.append(q_mask)
                if readout is not None:
                    sequence_parts.append(readout)
                    mask_parts.append(torch.ones(readout.shape[:2], dtype=torch.bool, device=device))
                if answer_prefix_ids is not None:
                    header_ids_row = answer_prefix_ids[row_index]
                    header_valid = answer_prefix_mask[row_index].bool()
                    header_width = int(header_valid.sum(dim=1).max())
                    header_ids = header_ids_row.new_full((count, header_width), int(tokenizer.pad_token_id))
                    header_mask = header_valid.new_zeros((count, header_width))
                    for slot in range(count):
                        real = header_ids_row[slot][header_valid[slot]]
                        header_ids[slot, header_width - real.numel():] = real
                        header_mask[slot, header_width - real.numel():] = True
                    sequence_parts.append(embedding(header_ids).to(dtype=self.dtype))
                    mask_parts.append(header_mask)
                prefill_mask = torch.cat(mask_parts, dim=1)
                prefill_counts = prefill_mask.sum(dim=1)
                sequence = torch.cat(sequence_parts, dim=1)
                if sequence.size(1) > self.window_limit:
                    raise ValueError("memory + prompt + readout + answer header exceeds the backbone window")
                sequence_positions = compact_positions(prefill_mask, 0)
                mask = build_causal_mask(prefill_mask, sequence.dtype)
                has_memory_cache = prefix.past_key_values is not None
                cache = self._copy_memory_cache(prefix, qa_context_indices[row_index])
                new_sequence = sequence[:, prefix.memory_length:] if has_memory_cache else sequence
                new_mask = prefill_mask if has_memory_cache else mask
                new_positions = (
                    sequence_positions[:, prefix.memory_length:]
                    if has_memory_cache else sequence_positions
                )
                output = self.qwen(
                    inputs_embeds=new_sequence,
                    attention_mask=new_mask,
                    position_ids=new_positions,
                    past_key_values=cache,
                    use_cache=True, return_dict=True,
                    logits_to_keep=1,
                )
                # Evaluation may run outside Accelerate's autocast context.  Keep the
                # complete embedding sequence in the frozen backbone dtype before it
                # enters Qwen/LoRA; otherwise a float32 readout or adapter branch can
                # reach a bf16 linear layer and fail with ``mat1 and mat2`` dtype errors.
                sequence = sequence.to(dtype=self.dtype)
                cache = output.past_key_values
                next_id = output.logits[:, -1].argmax(dim=-1)
                finished = torch.zeros(count, dtype=torch.bool, device=device)
                # Tokens are recorded in one padded buffer instead of a per-row Python list:
                # the step loop then does vectorised assignments rather than one append per
                # (row, step), and finished rows are excluded by the same mask that decides
                # what gets recorded.
                generated = torch.full(
                    (count, max_new_tokens), int(tokenizer.pad_token_id),
                    dtype=torch.long, device=device,
                )
                lengths = torch.zeros(count, dtype=torch.long, device=device)
                # Padding in either chat block remains blocked as generation appends tokens.
                # Rows without padding make the whole thing unnecessary -- which is the common
                # evaluation case, because a batch of contexts without padding has none either.
                cache_prefix_mask = prefill_mask
                prefill_width = cache_prefix_mask.size(1)
                needs_mask = not bool(cache_prefix_mask.all())
                if needs_mask:
                    # Step s needs `prefill + s + 1` columns, so concatenating a fresh mask
                    # every step copies the prefix O(steps) times. Allocate once and slice.
                    step_mask = torch.ones(
                        (count, prefill_width + max_new_tokens),
                        dtype=torch.bool, device=device,
                    )
                    step_mask[:, :prefill_width] = cache_prefix_mask
                for step in range(max_new_tokens):
                    active = ~finished
                    generated[:, step] = torch.where(active, next_id, generated[:, step])
                    lengths += active.long()
                    if eos_id is not None:
                        finished |= next_id.eq(int(eos_id))
                    if bool(finished.all()) or step + 1 == max_new_tokens:
                        break
                    # Finished rows are still fed a token to keep one batch shape; their
                    # outputs are ignored (only `active` rows are recorded) and positions are
                    # per-row, so they cannot disturb the rows still decoding.
                    token = embedding(next_id[:, None]).to(dtype=self.dtype)
                    position_ids = (prefill_counts + step).unsqueeze(1)
                    attention_mask = (
                        step_mask[:, :prefill_width + step + 1] if needs_mask else None
                    )
                    output = self.qwen(inputs_embeds=token, attention_mask=attention_mask,
                                       position_ids=position_ids, past_key_values=cache,
                                       use_cache=True, return_dict=True)
                    cache = output.past_key_values
                    next_id = output.logits[:, -1].argmax(dim=-1)
                result[row_index, :max_new_tokens] = generated
                all_lengths[row_index] = lengths

        width_out = int(all_lengths.max()) if n_rows else 0
        return result[:, :width_out]

    @torch.no_grad()
    def generate_answer_with_prefix(
        self, prefix, qa_context_indices, question_ids, question_mask,
        tokenizer, max_new_tokens=128, question_prefix_ids=None, question_prefix_mask=None,
        answer_prefix_ids=None, answer_prefix_mask=None,
    ):
        """Row-by-row decoding. Kept for callers that want the original behaviour."""
        return self.generate_answers_with_prefix(
            prefix, qa_context_indices, question_ids, question_mask, tokenizer,
            max_new_tokens=max_new_tokens, group_by_context=False,
            question_prefix_ids=question_prefix_ids, question_prefix_mask=question_prefix_mask,
            answer_prefix_ids=answer_prefix_ids, answer_prefix_mask=answer_prefix_mask,
        )

    @torch.no_grad()
    def generate_answer(self, context_ids, question_ids, context_mask, question_mask, tokenizer, max_new_tokens=128):
        embedding = self.qwen.get_input_embeddings()
        prefix = self.encode_context_prefix(embedding(context_ids), context_mask)
        indices = torch.arange(question_ids.size(0), device=question_ids.device)
        return self.generate_answer_with_prefix(prefix, indices, question_ids, question_mask, tokenizer, max_new_tokens)

    @torch.no_grad()
    def start_conversation(
        self, context_embeds, context_mask,
    ):
        """Create a single-session cache containing only the reusable memory prefix."""
        if not self.conversation_enabled:
            raise RuntimeError("memory.conversation.enabled must be true")
        if context_embeds.ndim != 3 or context_embeds.size(0) != 1:
            raise ValueError("conversation context must have batch size 1")
        prefix = self.encode_context_prefix(context_embeds, context_mask)
        memory = prefix.memory_embedding.to(dtype=self.dtype)
        return self._initialize_conversation_cache(prefix, memory)

    @torch.no_grad()
    def _initialize_conversation_cache(self, prefix, memory):
        return ConversationState(
            memory, prefix.memory_hidden, prefix.past_key_values,
            torch.empty(0, dtype=torch.long, device=memory.device),
        )

    def _copy_memory_cache(self, prefix, indices):
        """Fork the immutable context cache before extending it with a question."""
        if prefix.past_key_values is None:
            return None
        from transformers.cache_utils import DynamicCache

        indices = indices.to(prefix.memory_embedding.device, dtype=torch.long)
        return DynamicCache(
            ((layer.keys.index_select(0, indices), layer.values.index_select(0, indices))
             for layer in prefix.past_key_values.layers),
            config=self.qwen.config,
        )

    @torch.no_grad()
    def append_conversation_turn(self, state, turn_ids, tokenizer, max_new_tokens=128):
        """Append one rendered user turn, recompressing old memory/history at the limit.

        ``turn_ids`` is a rendered user turn ending in ``user_end + newline + assistant_header``.
        Only its current user body is encoded for the question resampler; prior history is never
        folded into the readout.
        """
        if not self.conversation_enabled:
            raise RuntimeError("memory.conversation.enabled must be true")
        if state.memory.ndim != 3 or state.memory.size(0) != 1:
            raise ValueError("conversation state must have batch size 1")
        if turn_ids.ndim == 1:
            turn_ids = turn_ids.unsqueeze(0)
        if turn_ids.ndim != 2 or turn_ids.size(0) != 1 or turn_ids.size(1) == 0:
            raise ValueError("turn_ids must contain one non-empty rendered user turn")
        turn_ids = turn_ids.to(state.memory.device, dtype=torch.long)
        user_ids = list(tokenizer.encode("<|im_start|>user\n", add_special_tokens=False))
        end_ids = list(tokenizer.encode("<|im_end|>", add_special_tokens=False))
        assistant_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        if not user_ids or not end_ids or not assistant_ids:
            raise ValueError("tokenizer must encode chat user/end markers and assistant header")
        row = turn_ids[0].tolist()
        user_role_ids = list(tokenizer.encode("<|im_start|>user", add_special_tokens=False))
        if row[:len(user_ids)] != user_ids:
            if not user_role_ids or row[:len(user_role_ids)] != user_role_ids:
                raise ValueError("rendered turn must start with a user header")
            user_body_start = len(user_role_ids)
        else:
            user_body_start = len(user_ids)
        assistant_start = next(
            (i for i in range(len(row) - len(assistant_ids), -1, -1)
             if row[i:i + len(assistant_ids)] == list(assistant_ids)),
            -1,
        )
        boundary = next(
            (i for i in range(assistant_start - len(end_ids), -1, -1)
             if row[i:i + len(end_ids)] == end_ids),
            -1,
        )
        if assistant_start < 0 or boundary <= 0:
            raise ValueError("rendered turn must contain a non-empty question and user end marker")
        if boundary <= user_body_start:
            raise ValueError("rendered turn must contain a non-empty question")
        question_ids = turn_ids[:, user_body_start:boundary]
        user_turn_ids = turn_ids[:, :boundary]
        closing_ids = turn_ids[:, boundary:]
        readout_length = self.readout_length if self.resampler is not None else 0
        occupied = (
            state.memory.size(1) + state.history_ids.numel() + turn_ids.size(1)
        )
        initial_window_need = (
            state.past_key_values.get_seq_length() + user_turn_ids.size(1)
            + readout_length + closing_ids.size(1) + max(0, int(max_new_tokens))
        )
        if occupied >= self.recompress_threshold or initial_window_need > self.window_limit:
            history_budget = self.max_recompress_tokens - state.memory.size(1)
            history = state.history_ids[-history_budget:] if history_budget else state.history_ids[:0]
            history_embeds = self.qwen.get_input_embeddings()(history)
            compression_input = torch.cat([state.memory, history_embeds[None]], dim=1)
            compression_mask = torch.ones(
                compression_input.shape[:2], dtype=torch.bool, device=compression_input.device
            )
            compressed = self.encode_context_prefix(
                compression_input, compression_mask, max_tokens=self.max_recompress_tokens
            )
            state.memory = compressed.memory_embedding
            state.history_ids = state.history_ids.new_empty((0,))
            refreshed = self._initialize_conversation_cache(
                compressed, state.memory,
            )
            state.memory_hidden = refreshed.memory_hidden
            state.past_key_values = refreshed.past_key_values

        embedding = self.qwen.get_input_embeddings()
        question_embeds = embedding(user_turn_ids).to(dtype=self.dtype)
        question_body_embeds = embedding(question_ids).to(dtype=self.dtype)
        cache_length = state.past_key_values.get_seq_length()
        input_length = user_turn_ids.size(1)
        required_length = (
            cache_length + input_length + readout_length + closing_ids.size(1)
            + max(0, int(max_new_tokens))
        )
        if required_length > self.window_limit:
            raise ValueError("conversation turn and generation budget exceed the backbone window")
        attention_mask = torch.ones(
            (1, cache_length + input_length), dtype=torch.bool, device=turn_ids.device
        )
        positions = torch.arange(cache_length, cache_length + input_length, device=turn_ids.device)[None]
        with self.lora_enabled(False):
            output = self.qwen(
                inputs_embeds=question_embeds, attention_mask=attention_mask,
                position_ids=positions, past_key_values=state.past_key_values,
                use_cache=True, output_hidden_states=True, return_dict=True,
                logits_to_keep=1,
            )
        state.past_key_values = output.past_key_values
        if self.resampler is not None:
            question_body_mask = torch.ones(
                question_body_embeds.shape[:2], dtype=torch.bool, device=question_body_embeds.device
            )
            if self.readout_mode == "learned_prefix":
                readout = self.resampler(question_body_embeds, question_body_mask, None)
            elif self.question_resampler_info_source == "embedding":
                readout = self.resampler(
                    torch.cat([state.memory, question_body_embeds], dim=1),
                    torch.cat([
                        torch.ones(state.memory.shape[:2], dtype=torch.bool, device=state.memory.device),
                        question_body_mask,
                    ], dim=1),
                )
            else:
                question_hidden = self._encode_independent_question_hidden(
                    question_body_embeds, question_body_mask
                )
                readout_hidden = torch.cat([state.memory_hidden, question_hidden], dim=1)
                readout_mask = torch.ones(
                    readout_hidden.shape[:2], dtype=torch.bool, device=readout_hidden.device
                )
                readout = self.resampler(readout_hidden, readout_mask).to(dtype=self.dtype)
            question_prefix = readout
        else:
            question_prefix = question_embeds[:, :0]
        closing_embeds = embedding(closing_ids).to(dtype=self.dtype)
        tail = torch.cat([question_prefix, closing_embeds], dim=1)
        tail_length = tail.size(1)
        attention_mask = torch.ones(
            (1, cache_length + input_length + tail_length),
            dtype=torch.bool, device=turn_ids.device,
        )
        positions = torch.arange(
            cache_length + input_length, cache_length + input_length + tail_length,
            device=turn_ids.device,
        )[None]
        with self.lora_enabled(False):
            output = self.qwen(
                inputs_embeds=tail, attention_mask=attention_mask,
                position_ids=positions, past_key_values=state.past_key_values,
                use_cache=True, return_dict=True, logits_to_keep=1,
            )
        state.past_key_values = output.past_key_values
        next_id = output.logits[:, -1].argmax(dim=-1)
        decode_cache_length = cache_length + input_length + tail_length
        generated = []
        eos_id = getattr(tokenizer, "eos_token_id", None)
        for _ in range(max(0, int(max_new_tokens))):
            token_id = int(next_id.item())
            generated.append(token_id)
            token = embedding(next_id[:, None])
            attention_mask = torch.ones(
                (1, decode_cache_length + 1), dtype=torch.bool, device=turn_ids.device
            )
            position = torch.tensor([[decode_cache_length]], device=turn_ids.device)
            with self.lora_enabled(False):
                output = self.qwen(
                    inputs_embeds=token, attention_mask=attention_mask,
                    position_ids=position, past_key_values=state.past_key_values,
                    use_cache=True, return_dict=True,
                )
            state.past_key_values = output.past_key_values
            decode_cache_length += 1
            if eos_id is not None and token_id == int(eos_id):
                break
            next_id = output.logits[:, -1].argmax(dim=-1)
        response_ids = torch.tensor(generated, dtype=torch.long, device=turn_ids.device)
        # Completed turns follow the native history template: no generation-only think
        # preamble or old readout. Re-encode the turn because its answer KV depended on both.
        response_body = generated
        if eos_id is not None and response_body and response_body[-1] == int(eos_id):
            response_body = response_body[:-1]
        question_text = tokenizer.decode(question_ids[0].tolist(), skip_special_tokens=False)
        if user_body_start != len(user_ids):
            question_text = question_text.removeprefix("\n")
        completed = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": question_text},
                {"role": "assistant", "content": tokenizer.decode(response_body, skip_special_tokens=False)},
                # Qwen3 treats even a completed final assistant as the active turn.
                # A following user makes it render the assistant as historical content.
                {"role": "user", "content": ""},
            ],
            tokenize=False, add_generation_prompt=False,
        )
        completed_tokens = tokenizer.encode(completed, add_special_tokens=False)
        trailer = tokenizer.encode("<|im_start|>user\n<|im_end|>\n", add_special_tokens=False)
        if not trailer or completed_tokens[-len(trailer):] != trailer:
            raise ValueError("completed history template must end with the empty user turn")
        completed_ids = torch.tensor(
            completed_tokens[:-len(trailer)],
            dtype=torch.long, device=turn_ids.device,
        )
        cached_length = state.past_key_values.get_seq_length()
        state.past_key_values.crop(-(cached_length - cache_length))
        with self.lora_enabled(False):
            output = self.qwen(
                inputs_embeds=embedding(completed_ids[None]).to(dtype=self.dtype),
                attention_mask=torch.ones(
                    (1, cache_length + completed_ids.numel()), dtype=torch.bool, device=turn_ids.device,
                ),
                position_ids=torch.arange(
                    cache_length, cache_length + completed_ids.numel(), device=turn_ids.device,
                )[None],
                past_key_values=state.past_key_values,
                use_cache=True, return_dict=True, logits_to_keep=1,
            )
        state.past_key_values = output.past_key_values
        state.history_ids = torch.cat([state.history_ids, completed_ids])
        state.turn_count += 1
        return response_ids


def recon_task_token_ids(tokenizer) -> tuple[int, ...]:
    """Token ids of the user-side Recon instruction, resolved from a tokenizer.

    The mean embedding of ``Reconstruct the context.`` initialises the trainable marker.
    Resolving ids with the active tokenizer avoids hardcoding vocabulary-specific rows; an empty
    result leaves the marker randomly initialized.
    """
    try:
        ids = tokenizer.encode("Reconstruct the context.", add_special_tokens=False)
    except (AttributeError, TypeError, ValueError):
        return ()
    return tuple(int(value) for value in (ids or ()))


def load_model(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from utils.config import dtype_from_name
    cfg.validate()
    # One tokenizer, and a pad token even for checkpoints whose config has none.
    tok = AutoTokenizer.from_pretrained(cfg.model.name_or_path, local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    from .data import split_chat_prompt
    recon_prompt = tok.apply_chat_template(
        [{"role": "user", "content": ""}], tokenize=False,
        add_generation_prompt=True, enable_thinking=cfg.data.chat_template_enable_thinking,
    )
    recon_user, _, recon_answer = split_chat_prompt(recon_prompt)
    qwen=AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path, dtype=dtype_from_name(cfg.model.torch_dtype),
        attn_implementation=cfg.model.attn_implementation, local_files_only=True,
    )
    qwen.config.use_cache=False
    model=ContextMemoryModel(
        qwen, rank=cfg.model.lora_rank, alpha=cfg.model.lora_alpha,
        memory_length=cfg.memory.memory_length,
        target_modules=cfg.model.target_modules, dropout=cfg.model.lora_dropout,
        max_context_tokens=cfg.data.max_context_tokens,
        trainable_dtype=dtype_from_name(cfg.model.trainable_dtype),
        context_gradient_checkpointing=cfg.model.context_gradient_checkpointing,
        gradient_checkpoint_every_n_layers=cfg.training.gradient_checkpoint_every_n_layers,
        readout_length=cfg.memory.readout_length,
        readout_mode=cfg.memory.readout_mode,
        question_resampler_layers=cfg.memory.question_resampler_layers,
        question_resampler_heads=cfg.memory.question_resampler_heads,
        question_resampler_hidden_size=cfg.memory.question_resampler_hidden_size,
        question_resampler_ffn_ratio=cfg.memory.question_resampler_ffn_ratio,
        question_resampler_use_self_attention=cfg.memory.question_resampler_use_self_attention,
        context_layer_indices=cfg.memory.context_layer_indices,
        memory_resampler_layers=cfg.memory.memory_resampler_layers,
        memory_resampler_heads=cfg.memory.memory_resampler_heads,
        memory_resampler_hidden_size=cfg.memory.memory_resampler_hidden_size,
        memory_resampler_ffn_ratio=cfg.memory.memory_resampler_ffn_ratio,
        memory_resampler_use_self_attention=cfg.memory.memory_resampler_use_self_attention,
        question_resampler_info_source=cfg.memory.question_resampler_info_source,
        conversation_enabled=cfg.memory.conversation.enabled,
        recompress_threshold=cfg.memory.conversation.recompress_threshold,
        max_recompress_tokens=cfg.memory.conversation.max_recompress_tokens,
        readout_keep_question_tokens=cfg.memory.readout_keep_question_tokens,
        task_init_token_ids=recon_task_token_ids(tok),
        recon_user_token_ids=tuple(tok(recon_user, add_special_tokens=False).input_ids),
        recon_answer_prefix_token_ids=tuple(tok(recon_answer, add_special_tokens=False).input_ids),
    )
    for name,p in model.named_parameters():
        # Preserve module-level freezing while enforcing the global trainable-name policy.
        p.requires_grad = p.requires_grad and is_trainable_parameter_name(name)
    return tok, model


__all__ = [
    "ContextMemoryModel", "ContextMemoryOutput", "ConversationState",
    "MetaLoRA", "ContextPrefix",
    "ContextMemoryResampler", "QuestionResampler",
    "build_causal_mask", "compact_positions",
    "is_trainable_parameter_name", "load_model", "TASK_RECON", "TASK_QA",
]
