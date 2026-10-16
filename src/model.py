from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .dtypes import no_autocast
from .resampler import ContextMemoryResampler, LearnedPrefix, QuestionResampler

try:
    from transformers.cache_utils import DynamicCache
except ImportError:  # pragma: no cover - transformers is required at runtime
    DynamicCache = None


@dataclass
class MetaLoRAOutput:
    logits: torch.Tensor | None
    labels: torch.LongTensor
    memory: torch.Tensor
    layer_memory: torch.Tensor
    supervision_losses: torch.Tensor | None = None
    # Which supervision rows carried a target (see losses.supervision_objectives).
    supervision_mask: torch.Tensor | None = None
    # Per-row (nll, correct, tokens) of the teacher-forced reconstruction, when requested.
    row_stats: dict | None = None
    objective_terms: dict | None = None



@dataclass
class ContextPrefix:
    layer_memory: torch.Tensor
    memory: torch.Tensor
    memory_cache: object
    context_length: int
    memory_length: int
    # Last-layer states of context rows, retained only when distillation is enabled.
    context_hidden: torch.Tensor | None = None
    memory_embedding: torch.Tensor | None = None
    layer_memory_embedding: torch.Tensor | None = None


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


class TiedUnembedding(nn.Module):
    """Zero-parameter unembedding: score with the backbone's own (tied) embedding.

    ``tie_word_embeddings`` makes the language-model head the input embedding matrix.
    Distillation reuses that frozen unembedding and introduces no trainable parameters.
    """

    def __init__(self, embedding_getter):
        super().__init__()
        if embedding_getter is None:
            raise ValueError("TiedUnembedding needs an embedding getter")
        self._embedding_getter = embedding_getter

    @property
    def compute_dtype(self) -> torch.dtype:
        return self._embedding_getter().dtype

    def materialized_weight(self) -> torch.Tensor:
        return self._embedding_getter()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        with no_autocast(hidden.device):
            return F.linear(hidden, self.materialized_weight())



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


def build_continuation_mask(
    question_mask: torch.Tensor,
    answer_mask: torch.Tensor,
    memory_length: int,
    dtype: torch.dtype,
    readout_length: int = 0,
) -> torch.Tensor:
    """Mask for question/read-out/answer tokens attending to a memory-only cache.

    The optional read-out rows sit between the question and the answer. They are built from
    the question and the memory, so they may read every valid question key but never the
    answer; the answer rows may read them.
    """
    question_mask, answer_mask = question_mask.bool(), answer_mask.bool()
    bsz, question_length = question_mask.shape
    answer_length = answer_mask.shape[1]
    readout_length = int(readout_length)
    if readout_length < 0:
        raise ValueError("readout_length must be >= 0")
    current = question_length + readout_length + answer_length
    device = question_mask.device
    row = torch.arange(current, device=device)
    causal = row[:, None] >= row[None, :]
    allowed = torch.zeros((bsz, current, memory_length + current), dtype=torch.bool, device=device)
    allowed[:, :, :memory_length] = True
    r0 = question_length + readout_length
    allowed[:, :question_length, memory_length:memory_length + question_length] = (
        question_mask[:, None, :] & causal[:question_length, :question_length]
    )
    if readout_length:
        allowed[:, question_length:r0, memory_length:memory_length + question_length] = question_mask[:, None, :]
    allowed[:, r0:, memory_length:memory_length + question_length] = question_mask[:, None, :]
    if readout_length:
        allowed[:, r0:, memory_length + question_length:memory_length + r0] = True
    allowed[:, r0:, memory_length + r0:] = (
        answer_mask[:, None, :] & causal[r0:, r0:]
    )
    valid = torch.cat(
        [question_mask, question_mask.new_ones(bsz, readout_length), answer_mask], dim=1
    )
    self_edge = torch.zeros_like(allowed)
    self_edge[:, row, memory_length + row] = True
    allowed = torch.where(valid[:, :, None], allowed, self_edge)
    mask = torch.zeros((bsz, 1, current, memory_length + current), dtype=dtype, device=device)
    return mask.masked_fill(~allowed[:, None], torch.finfo(dtype).min)


class ContextMemoryModel(nn.Module):
    """Qwen encoder/decoder with hand-written MetaLoRA.

    Two dtypes coexist. The frozen Qwen backbone stays in its checkpoint dtype
    (``qwen_dtype``, bfloat16 for Qwen3) and every tensor that enters it is cast to
    that dtype. The trainable pieces -- learned memory queries, task embeddings and
    ``lora_A``/``lora_B`` -- can be kept in ``trainable_dtype`` (float32 by
    default) so that AdamW's moments and its ``param.add_(update, alpha=-lr)`` step
    are not quantised to bfloat16 resolution. The casts that reconcile the two dtypes
    live in the context resampler and short memory pass,
    ``MetaLoRA`` casts on entry and exit, and the losses
    upcast before reducing.
    """
    def __init__(
        self, qwen: nn.Module, rank=8, alpha=16.0, memory_length=8,
        target_modules=None, dropout=0.0, max_context_tokens=None,
        trainable_dtype=torch.float32, distillation=False,
        task_init_token_ids: tuple[int, ...] | None = None,
        readout_length=0, readout_layers=2, readout_heads=4, readout_hidden_size=256,
        readout_mode="memory_question", context_layer_indices=(-1,),
        memory_resampler_layers=2, memory_resampler_heads=4,
        memory_resampler_hidden_size=256, memory_resampler_ffn_ratio=2,
        memory_resampler_output_mode="layer0",
        reader_memory_source="memory_hidden", context_gradient_checkpointing=False,
        gradient_checkpoint_every_n_layers=1,
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
        if not 0 < memory_length <= self.window_limit:
            raise ValueError("memory_length must be within the backbone window")
        self.context_gradient_checkpointing = bool(context_gradient_checkpointing)
        if type(gradient_checkpoint_every_n_layers) is not int or gradient_checkpoint_every_n_layers <= 0:
            raise ValueError(
                "gradient_checkpoint_every_n_layers must be a positive integer"
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
            backbone_layers=depth,
            output_mode=memory_resampler_output_mode,
        )
        # Recon has one continuous task marker. It is initialised from the mean of the
        # pretrained input embeddings of the assistant generation header, which gives it the
        # same role and scale as the chat-template prefix while keeping it a trainable vector.
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
        # Distillation reuses a memory-prefixed context pass and the frozen tied
        # unembedding; it carries no trainable parameters of its own.
        self.distillation = bool(distillation)
        self.distill_head = (
            TiedUnembedding(lambda: self.qwen.get_input_embeddings().weight)
            if self.distillation else None
        )
        # Per-question read-out over the (question-agnostic, cacheable) memory. Zero means
        # the historical behaviour: the answer attends only to the memory and the question.
        if readout_mode not in {"memory_question", "memory_only", "learned_prefix", "off"}:
            raise ValueError(f"invalid readout_mode: {readout_mode}")
        if readout_length < 0:
            raise ValueError("readout_length must be non-negative")
        self.readout_mode = readout_mode
        if reader_memory_source not in {"memory_hidden", "memory_embedding", "layer_mean"}:
            raise ValueError("reader_memory_source must be memory_hidden, memory_embedding or layer_mean")
        self.reader_memory_source = reader_memory_source
        self.readout_length = 0 if readout_mode == "off" else int(readout_length)
        self.resampler = None
        if self.readout_length > 0:
            if readout_mode == "learned_prefix":
                self.resampler = LearnedPrefix(self.qwen_hidden_size, self.readout_length, dtype=trainable_dtype)
            else:
                self.resampler = QuestionResampler(
                    self.qwen_hidden_size, self.readout_length, readout_layers, readout_heads,
                    width=readout_hidden_size, dtype=trainable_dtype, mode=readout_mode,
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

        The long context encoder keeps LoRA enabled by default. Reader passes that
        consume the resampler output use ``lora_enabled(False)`` so they match the
        original frozen Qwen used at inference.
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

    def _memory_prefix(self, layer_memory):
        return layer_memory[:, -1]  # final Qwen layer memory is the input prefix

    @property
    def _transformer_body(self):
        """The decoder stack without the language-model head, resolved once and cached.

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
            self.context_gradient_checkpointing and self.training and torch.is_grad_enabled()
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
            selected.append(captured[index])
        weights = self.context_layer_logits.softmax(dim=0)
        mixed = sum(
            weight.to(hidden.dtype) * hidden
            for weight, hidden in zip(weights, selected, strict=True)
        )
        return mixed, out.last_hidden_state

    def encode_context_prefix(self, context_embeds, context_mask) -> ContextPrefix:
        """Encode context once, resample it, then run one short memory-only pass."""
        if context_embeds.ndim != 3 or context_embeds.size(-1) != self.qwen_hidden_size:
            raise ValueError("context_embeds must have shape [B, L, backbone_hidden_size]")
        if context_mask.shape != context_embeds.shape[:2]:
            raise ValueError("context_mask must have shape [B, L]")
        if not 0 < context_embeds.size(1) <= self.max_context_tokens:
            raise ValueError(
                f"context length {context_embeds.size(1)} must be within the configured "
                f"max_context_tokens ({self.max_context_tokens}); the backbone window is "
                f"{self.window_limit}"
            )
        if not bool(context_mask.bool().any(dim=1).all()):
            raise ValueError("every context must contain at least one valid token")
        context_embeds = context_embeds.to(dtype=self.dtype)
        context_mask = context_mask.bool()
        # One standard causal Qwen pass over the context. The context KV is not cached.
        # Rank-based positions make the pass independent of the padding side: a right-padded
        # and a left-padded batch number the same real tokens the same way.
        positions = compact_positions(context_mask, 0)
        with self.lora_enabled(True):
            context_hidden, context_last_hidden = self._encode_context_hidden(
                context_embeds, context_mask, positions,
            )
        encoded_memory = self.context_resampler(context_hidden, context_mask).to(dtype=self.dtype)
        if self.context_resampler.output_mode == "layer0":
            memory_embedding = encoded_memory
            layer_memory_embedding = None
        else:
            layer_memory_embedding = encoded_memory
            memory_embedding = layer_memory_embedding[:, 0]
        # Short Qwen pass over fixed-length memory embeddings produces the reusable KV.
        memory_positions = torch.arange(memory_embedding.size(1), device=memory_embedding.device)
        memory_positions = memory_positions.unsqueeze(0).expand(memory_embedding.size(0), -1)
        memory_mask = torch.ones(
            memory_embedding.size(0), memory_embedding.size(1),
            dtype=torch.bool, device=memory_embedding.device,
        )
        body = self._transformer_body
        hooks = []
        layers = getattr(body, "layers", None)
        if layers is None:
            raise RuntimeError("Qwen transformer body does not expose decoder layers")
        if layer_memory_embedding is not None and layer_memory_embedding.size(1) != len(layers):
            raise RuntimeError("layer-wise memory embedding count does not match Qwen layers")

        def inject(layer_index):
            def hook(_module, inputs):
                hidden = inputs[0]
                conditioned = hidden.clone()
                conditioned[:, :memory_embedding.size(1)] = (
                    hidden[:, :memory_embedding.size(1)]
                    + self.context_resampler.layer_gates[layer_index]
                    * layer_memory_embedding[:, layer_index]
                )
                return (conditioned, *inputs[1:])
            return hook

        if layer_memory_embedding is not None:
            for index, layer in enumerate(layers):
                hooks.append(layer.register_forward_pre_hook(inject(index)))
        try:
            with self.lora_enabled(False):
                memory_out = body(
                    inputs_embeds=memory_embedding,
                    attention_mask=memory_mask,
                    position_ids=memory_positions,
                    output_hidden_states=True,
                    use_cache=True,
                    return_dict=True,
                )
        finally:
            for hook in hooks:
                hook.remove()
        layer_memory = torch.stack(memory_out.hidden_states[1:], dim=1)
        cache = memory_out.past_key_values
        if cache is None:
            raise RuntimeError("Qwen did not return a KV cache for prefix encoding")
        return ContextPrefix(
            layer_memory, self._memory_prefix(layer_memory), cache,
            context_embeds.size(1), memory_embedding.size(1),
            context_hidden=context_last_hidden if self.distillation else None,
            memory_embedding=memory_embedding,
            layer_memory_embedding=layer_memory_embedding,
        )

    @staticmethod
    def _select_cache(cache, indices, config):
        """A cache holding only the rows ``indices`` of ``cache``.

        ``DynamicCache(ddp_cache_data=..., config=...)`` is the transformers 5.x constructor
        (requirements.lock pins the version we test). A different major version fails with a
        TypeError that says nothing about the cause, so translate it here.
        """
        layers = [
            (layer.keys.index_select(0, indices), layer.values.index_select(0, indices))
            for layer in cache.layers
        ]
        try:
            return DynamicCache(ddp_cache_data=layers, config=config)
        except TypeError as error:
            import transformers

            raise RuntimeError(
                "this build expects the transformers 5.x "
                "DynamicCache(ddp_cache_data=..., config=...) constructor, which "
                f"transformers {transformers.__version__} does not provide; the tested "
                "versions are in requirements.lock"
            ) from error

    def _readout_embeddings(self, prefix, question_embeds, question_mask, indices=None):
        """``[B, R, H]`` question-conditioned positions, or None when the read-out is off.

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
        if self.reader_memory_source == "memory_embedding":
            source = prefix.memory_embedding
        elif self.reader_memory_source == "layer_mean":
            source = prefix.layer_memory.mean(dim=1)
        else:
            source = prefix.memory
        memory = source if indices is None else source.index_select(0, indices)
        readout = self.resampler(question_embeds, question_mask, memory)
        return readout.to(dtype=self.dtype)

    def _qa_positions(self, prefix, question_mask, answer_length, readout_length, device):
        """Absolute positions of ``[question, (read-out), answer]`` after memory.

        The numbering is the one the generation path builds row by row
        (:meth:`generate_answers_with_prefix`): every real token takes the rank of its own
        slot, so a row's real question tokens sit at ``start + i``, the read-out slots at
        ``start + valid + j`` and the answer at ``start + valid + readout_length + k``,
        whatever the question padding looks like. Padding slots are masked out, so their
        position never reaches attention. It used to be a single ``arange`` over the padded
        block, which made the memory-to-question distance depend on the longest question in
        the batch and disagreed with generation.

        Both blocks are numbered by :func:`compact_positions`, so the equivalence with
        generation is structural rather than something the tests have to keep re-checking.
        """
        question_mask = question_mask.bool()
        blocks = [question_mask]
        if readout_length:
            blocks.append(
                question_mask.new_ones(question_mask.size(0), int(readout_length))
            )
        if answer_length:
            # The answer block is padded on the right and its padding is dropped by the
            # label mask, so all of its slots are numbered as real ones -- exactly what a
            # row-by-row decode produces.
            blocks.append(
                question_mask.new_ones(question_mask.size(0), int(answer_length))
            )
        return compact_positions(torch.cat(blocks, dim=1), int(prefix.memory_length)).to(device)

    def forward_with_memory_prefix(
        self, prefix, qa_context_indices, question_embeds, question_mask,
        answer_embeds, answer_mask, labels, loss_only=False, max_logits_rows=256,
    ):
        question_embeds, answer_embeds = question_embeds.to(dtype=self.dtype), answer_embeds.to(dtype=self.dtype)
        indices = qa_context_indices.to(question_embeds.device, dtype=torch.long)
        question_mask = question_mask.bool()
        cache = self._select_cache(prefix.memory_cache, indices, self.qwen.config)
        readout = self._readout_embeddings(prefix, question_embeds, question_mask, indices)
        extra = 0 if readout is None else readout.size(1)
        if readout is None:
            sequence = torch.cat([question_embeds, answer_embeds], dim=1)
        else:
            sequence = torch.cat([question_embeds, readout, answer_embeds], dim=1)
        mask = build_continuation_mask(
            question_mask, answer_mask, prefix.memory_length, sequence.dtype,
            readout_length=extra,
        )
        positions = self._qa_positions(prefix, question_mask, answer_embeds.size(1), extra, sequence.device)
        if int(positions.max()) >= self.window_limit:
            raise ValueError("memory + prompt + readout + target exceeds the backbone window")
        body = self._transformer_body if loss_only else self.qwen
        with self.lora_enabled(False):
            out = body(
                inputs_embeds=sequence, attention_mask=mask, position_ids=positions,
                past_key_values=cache, use_cache=not loss_only, return_dict=True,
            )
        ignored = torch.full(
            (labels.size(0), question_embeds.size(1) + extra), -100,
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
        return MetaLoRAOutput(
            logits=None if loss_only else out.logits,
            labels=full_labels,
            memory=prefix.memory,
            layer_memory=prefix.layer_memory,
            supervision_losses=row_losses,
            supervision_mask=row_mask,
        )

    def forward_recon_with_memory_prefix(
        self, prefix, recon_context_indices, context_embeds, context_mask,
        labels, loss_only=False, max_logits_rows=256, return_stats=False,
    ):
        """Reconstruct context as assistant content after a memory/task user prefix."""
        context_embeds = context_embeds.to(dtype=self.dtype)
        context_mask = context_mask.bool()
        indices = recon_context_indices.to(context_embeds.device, dtype=torch.long)
        task = self.recon_task_embedding.to(dtype=self.dtype).view(1, 1, -1).expand(
            context_embeds.size(0), -1, -1
        )
        user_mask = torch.ones(
            (context_embeds.size(0), 1), dtype=torch.bool, device=context_embeds.device
        )
        cache = self._select_cache(prefix.memory_cache, indices, self.qwen.config)
        sequence = torch.cat([task, context_embeds], dim=1)
        mask = build_continuation_mask(
            user_mask, context_mask, prefix.memory_length, sequence.dtype,
        )
        positions = compact_positions(
            torch.cat([user_mask, context_mask], dim=1), int(prefix.memory_length)
        )
        if int(positions.max()) >= self.window_limit:
            raise ValueError("memory + recon task + context exceeds the backbone window")
        body = self._transformer_body if loss_only else self.qwen
        with self.lora_enabled(False):
            out = body(
                inputs_embeds=sequence,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=cache,
                use_cache=not loss_only,
                return_dict=True,
            )
        ignored = torch.full(
            (labels.size(0), 1), -100, dtype=labels.dtype, device=labels.device,
        )
        full_labels = torch.cat([ignored, labels], dim=1)
        row_losses = row_mask = row_stats = None
        if loss_only:
            from .losses import sequence_lm_loss, teacher_forced_row_stats
            head = self.qwen.get_output_embeddings()
            hidden, target = out.last_hidden_state[:, :-1], full_labels[:, 1:]
            row_losses, row_mask = sequence_lm_loss(
                hidden, target, head, max_logits_rows=max_logits_rows,
            )
            if return_stats:
                row_stats = teacher_forced_row_stats(
                    hidden, target, head, max_logits_rows=max_logits_rows,
                )
        return MetaLoRAOutput(
            logits=None if loss_only else out.logits,
            labels=full_labels,
            memory=prefix.memory,
            layer_memory=prefix.layer_memory,
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

    def _distillation_student_hidden(self, prefix, context_embeds, context_mask):
        """Run context tokens behind the reusable memory KV for distillation."""
        context_embeds = context_embeds.to(dtype=self.dtype)
        context_mask = context_mask.bool()
        if prefix.memory_length + context_embeds.size(1) > self.window_limit:
            raise ValueError("distillation needs memory_length + context_length within the backbone window")
        mask = torch.cat([
            context_mask.new_ones(context_mask.size(0), prefix.memory_length), context_mask,
        ], dim=1)
        positions = compact_positions(context_mask, int(prefix.memory_length))
        # Qwen updates a supplied DynamicCache even with use_cache=False. Keep
        # the context tokens out of the reusable memory-only prefix.
        indices = torch.arange(context_embeds.size(0), device=context_embeds.device)
        cache = self._select_cache(prefix.memory_cache, indices, self.qwen.config)
        with self.lora_enabled(False):
            out = self._transformer_body(
                inputs_embeds=context_embeds,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=cache,
                use_cache=False,
                return_dict=True,
            )
        return out.last_hidden_state

    def forward(
        self, context_embeds, context_mask, question_embeds, question_mask,
        answer_embeds, answer_mask, labels, qa_context_indices=None,
        context_ids=None, loss_config=None, loss_only=False,
        recon_batch=None, generator=None,
    ):
        """Training path.

        ``context_ids`` supplies targets for the optional distillation objective.
        ``loss_config`` is the objective's config section: it decides which auxiliary terms run
        and how many vocabulary rows are projected per chunk. ``generator`` makes the
        distillation position sampling recomputable (see ``losses.sample_positions``).
        """
        prefix = self.encode_context_prefix(context_embeds, context_mask)
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
            )
        else:
            output = MetaLoRAOutput(
                logits=None,
                labels=labels,
                memory=prefix.memory,
                layer_memory=prefix.layer_memory,
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
        if loss_config is not None:
            from .losses import memory_objectives
            output.objective_terms = memory_objectives(
                self, prefix, context_ids, context_mask, loss_config, generator=generator,
            )
            return output
        return output

    @torch.no_grad()
    def generate_answers_with_prefix(self, prefix, qa_context_indices, question_ids, question_mask,
                                     tokenizer, max_new_tokens=128, group_by_context=True,
                                     max_rows_per_group=4, task_id=TASK_QA):
        with self.lora_enabled(False):
            return self._generate_answers_with_prefix(
                prefix, qa_context_indices, question_ids, question_mask,
                tokenizer, max_new_tokens, group_by_context, max_rows_per_group,
                task_id,
            )

    def _generate_answers_with_prefix(self, prefix, qa_context_indices, question_ids, question_mask,
                                      tokenizer, max_new_tokens=128, group_by_context=True,
                                      max_rows_per_group=4, task_id=TASK_QA):
        """Greedy decoding for many QA rows that share one context prefix.

        Rows belonging to the same context share the same memory KV cache, so they are
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
        start = prefix.memory_length
        eos_id = getattr(tokenizer, "eos_token_id", None)
        n_rows = question_ids.size(0)

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
                valid = row_valid.sum(dim=1)                              # [count]
                # Recon rows have no question tokens; keep the user segment exactly
                # ``[memory, recon_task_embedding]`` instead of adding a masked pad.
                width = int(valid.max())
                # Left-align the real question tokens; compact_positions then restores each
                # row's own absolute positions, so the layout matches the row-by-row decode.
                q_ids = torch.full((count, width), int(tokenizer.pad_token_id), dtype=row_ids.dtype, device=device)
                q_mask = torch.zeros((count, width), dtype=torch.bool, device=device)
                for slot in range(count):
                    real = row_ids[slot][row_valid[slot]]
                    q_ids[slot, width - real.numel():] = real
                    q_mask[slot, width - real.numel():] = True
                question = embedding(q_ids).to(dtype=self.dtype)
                if task_id == TASK_RECON:
                    task = self.recon_task_embedding.to(dtype=self.dtype).view(1, 1, -1).expand(
                        count, -1, -1
                    )
                    question = torch.cat([task, question], dim=1)
                    task_mask = torch.ones((count, 1), dtype=torch.bool, device=device)
                    q_mask = torch.cat([task_mask, q_mask], dim=1)

                cache = self._select_cache(
                    prefix.memory_cache,
                    qa_context_indices[row_index].to(device, dtype=torch.long),
                    self.qwen.config,
                )
                readout = self._readout_embeddings(
                    prefix, question, q_mask,
                    qa_context_indices[row_index].to(device, dtype=torch.long),
                )
                extra = 0 if readout is None else readout.size(1)
                # One helper numbers the whole prefill block: the read-out slots land right
                # after each row's real question tokens, exactly where a row-by-row decode
                # would put them, and the first generated token is then at ``start + real``.
                prefill_mask = (
                    torch.cat([q_mask, q_mask.new_ones(count, extra)], dim=1)
                    if extra
                    else q_mask
                )
                prefill_counts = prefill_mask.sum(dim=1)
                sequence = question if readout is None else torch.cat([question, readout], dim=1)
                sequence_positions = compact_positions(prefill_mask, start)
                # Evaluation may run outside Accelerate's autocast context.  Keep the
                # complete embedding sequence in the frozen backbone dtype before it
                # enters Qwen/LoRA; otherwise a float32 readout or adapter branch can
                # reach a bf16 linear layer and fail with ``mat1 and mat2`` dtype errors.
                sequence = sequence.to(dtype=self.dtype)
                empty = q_mask.new_zeros(count, 0)
                mask = build_continuation_mask(
                    q_mask, empty, prefix.memory_length, question.dtype,
                    readout_length=extra,
                )
                output = self.qwen(inputs_embeds=sequence, attention_mask=mask, position_ids=sequence_positions,
                                   past_key_values=cache, use_cache=True, return_dict=True)
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
                # Question padding must stay blocked for the incremental steps too: the cache
                # layout is [memory, question (with padding), read-out, generated...] and a
                # plain all-ones mask would expose the padding keys the prefill masked out.
                # Rows without padding make the whole thing unnecessary -- which is the common
                # evaluation case, because a batch of contexts without padding has none either.
                cache_prefix_mask = torch.cat(
                    [prefill_mask.new_ones(count, prefix.memory_length), prefill_mask], dim=1
                )
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
                    position_ids = (start + prefill_counts + step).unsqueeze(1)
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
        tokenizer, max_new_tokens=128,
    ):
        """Row-by-row decoding. Kept for callers that want the original behaviour."""
        return self.generate_answers_with_prefix(
            prefix, qa_context_indices, question_ids, question_mask, tokenizer,
            max_new_tokens=max_new_tokens, group_by_context=False,
        )

    @torch.no_grad()
    def generate_answer(self, context_ids, question_ids, context_mask, question_mask, tokenizer, max_new_tokens=128):
        embedding = self.qwen.get_input_embeddings()
        prefix = self.encode_context_prefix(embedding(context_ids), context_mask)
        indices = torch.arange(question_ids.size(0), device=question_ids.device)
        return self.generate_answer_with_prefix(prefix, indices, question_ids, question_mask, tokenizer, max_new_tokens)


def assistant_header_token_ids(tokenizer) -> tuple[int, ...]:
    """Token ids of the assistant generation header, resolved from a tokenizer.

    ``<|im_start|>assistant\n`` is what a chat template emits before an answer, and the mean of
    its pretrained input embeddings initialises the Recon marker. Resolving the ids here instead
    of hardcoding them keeps a different Qwen family or a re-cut vocabulary from silently
    initialising that marker from unrelated rows; an empty result (the header is not in the
    vocabulary) leaves the task marker randomly initialized.
    """
    try:
        ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
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
    qwen=AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path, dtype=dtype_from_name(cfg.model.torch_dtype),
        attn_implementation=cfg.model.attn_implementation, local_files_only=True,
    )
    if cfg.loss.distill_weight > 0:
        window = int(qwen.config.max_position_embeddings)
        effective_context_tokens = min(
            cfg.data.max_context_tokens, cfg.data.max_input_tokens,
        )
        if cfg.memory.memory_length + effective_context_tokens > window:
            raise ValueError(
                "distillation requires "
                f"memory_length + effective input context length <= backbone window ({window})"
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
        distillation=cfg.loss.distill_weight > 0,
        readout_length=cfg.memory.readout_length,
        readout_mode=cfg.memory.readout_mode,
        readout_layers=cfg.memory.readout_layers,
        readout_heads=cfg.memory.readout_heads,
        readout_hidden_size=cfg.memory.readout_hidden_size,
        context_layer_indices=cfg.memory.context_layer_indices,
        memory_resampler_layers=cfg.memory.memory_resampler_layers,
        memory_resampler_heads=cfg.memory.memory_resampler_heads,
        memory_resampler_hidden_size=cfg.memory.memory_resampler_hidden_size,
        memory_resampler_ffn_ratio=cfg.memory.memory_resampler_ffn_ratio,
        memory_resampler_output_mode=cfg.memory.memory_resampler_output_mode,
        reader_memory_source=cfg.memory.reader_memory_source,
        task_init_token_ids=assistant_header_token_ids(tok),
    )
    for name,p in model.named_parameters():
        # Preserve module-level freezing (for example the inactive layer-0
        # projection in ``layerwise`` resampler mode) while enforcing the global
        # trainable-name policy.
        p.requires_grad = p.requires_grad and is_trainable_parameter_name(name)
    return tok, model


__all__ = [
    "ContextMemoryModel", "MetaLoRAOutput",
    "MetaLoRA", "ContextPrefix",
    "TiedUnembedding",
    "ContextMemoryResampler", "QuestionResampler",
    "build_continuation_mask", "compact_positions",
    "is_trainable_parameter_name", "load_model", "TASK_RECON", "TASK_QA",
]
