from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .MemoryDecoder import MemoryDecoder
from .dtypes import no_autocast
from .resampler import QuestionResampler

try:
    from transformers.cache_utils import DynamicCache
except ImportError:  # pragma: no cover - transformers is required at runtime
    DynamicCache = None


@dataclass
class MetaLoRAOutput:
    logits: torch.Tensor
    labels: torch.LongTensor
    memory: torch.Tensor
    layer_memory: torch.Tensor
    recon: torch.Tensor
    context_target: torch.Tensor
    context_mask: torch.Tensor
    # Context next-token prediction terms. Populated only when the auxiliary
    # objective is "token_recon"; ``recon``/``context_target`` are None then.
    token_recon_hidden: torch.Tensor | None = None
    token_recon_labels: torch.LongTensor | None = None
    token_recon_mask: torch.Tensor | None = None
    # Memory-prefixed teacher-forced recon of the context (the autoencoding
    # objective). Populated only when ``memory.causal_recon_weight`` is non-zero.
    objective_terms: dict | None = None
    ae_hidden: torch.Tensor | None = None
    ae_labels: torch.LongTensor | None = None
    ae_mask: torch.Tensor | None = None
    # The same positions scored by the *plain causal LM* (no memory): the distillation
    # target, and the honest reference for "what does the memory actually add".
    ae_teacher_hidden: torch.Tensor | None = None


@dataclass
class ContextLMTerms:
    """Sampled context next-token prediction terms, shared by train and eval."""
    hidden: torch.Tensor          # [B, num_layers, P, D] decoder hidden states
    labels: torch.LongTensor      # [B, P] target context token ids (-100 = ignore)
    mask: torch.Tensor            # [B, P] which of the P positions are real


@dataclass
class ContextPrefix:
    layer_memory: torch.Tensor
    memory: torch.Tensor
    memory_cache: object
    recon: torch.Tensor
    context_target: torch.Tensor
    context_mask: torch.Tensor
    context_length: int
    memory_length: int
    # Last-layer states of the *context* rows of the encoder pass. Those rows attend only
    # to earlier context tokens, so they are the plain causal-LM distribution for the
    # context -- i.e. the teacher a distillation objective needs, for free.
    context_hidden: torch.Tensor | None = None


def is_trainable_parameter_name(name: str) -> bool:
    """Single source of truth for which parameters are trained.

    Used both by :meth:`MetaLoRA.set_trainable_dtype` and by ``load_model`` so that a
    newly added trainable module cannot be forgotten in one of the two places.
    """
    return (
        "lora_A" in name
        or "lora_B" in name
        or name == "memory_tokens"
        or name.startswith(("decoders.", "token_recon_head.", "resampler."))
    )


class TiedUnembedding(nn.Module):
    """Zero-parameter unembedding: score with the backbone's own (tied) embedding.

    ``tie_word_embeddings`` makes the language-model head the input embedding matrix, and
    that is exactly the read-out the context-compression papers use for their
    autoencoding objective. Using it here introduces no parameters, so the autoencoding
    loss trains only the memory slots / LoRA adapters and never a fresh classifier; the
    memory-side ``token_recon_head`` stays dedicated to the memory-only probe objective.
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


class VocabularyHead(nn.Module):
    """Shared unembedding for the context next-token objective.

    ``mode="linear"`` (default) is the original behaviour: a from-scratch
    ``[vocab, D]`` weight matrix trained by the loss. ``mode="tied"`` instead reuses
    the backbone's own (frozen, tied) input embedding ``E`` and trains only a
    ``D -> H`` adapter, so the classifier lives in the pretrained token geometry:

        W = E @ adapter.weight            # [vocab, D]

    This is the read-out the compression papers rely on ("score with the model's own
    embedding"): a random 38.9M-parameter head is a cold start that has to learn the
    whole vocabulary geometry from 256-dim bottleneck states, and its gradient noise
    is what the memory tokens receive first.

    ``W`` is materialised once per parameter version and cached: the naive two-step
    form ``(h @ A.T) @ E.T`` costs ``H * vocab`` per row against ``D * vocab`` for the
    materialised one (8x for Qwen3-1.7B), while the materialisation itself is one
    ``vocab x H x D`` matmul (~3 ms) against ~86k scored rows per step -- the
    break-even point is ~292 rows, so materialising always wins here. The cache key is
    the adapter weight's version counter, so an optimizer step (an in-place update)
    invalidates it automatically and a stale head can never score a later forward;
    ``refresh()`` clears it explicitly. ``E`` is a frozen parameter of the backbone, so
    only the adapter receives gradient.
    """

    def __init__(self, hidden_size, vocab_size, mode="linear", embedding_getter=None,
                 backbone_hidden_size=None, init_adapter=None):
        super().__init__()
        if mode not in {"linear", "tied"}:
            raise ValueError(f"VocabularyHead mode must be 'linear' or 'tied', got {mode!r}")
        self.mode = mode
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self._embedding_getter = embedding_getter
        self._cached_weight = None
        self._cache_key = None
        if mode == "linear":
            # Same shape and same init as the nn.Linear this replaces, so parameter
            # names (token_recon_head.weight) and RNG consumption stay identical.
            self.weight = nn.Parameter(torch.empty(self.vocab_size, self.hidden_size))
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        else:
            if embedding_getter is None or backbone_hidden_size is None:
                raise ValueError("tied mode needs embedding_getter and backbone_hidden_size")
            self.adapter = nn.Linear(self.hidden_size, int(backbone_hidden_size), bias=False)
            if init_adapter is not None:
                expected = (int(backbone_hidden_size), self.hidden_size)
                if tuple(init_adapter.shape) != expected:
                    raise ValueError(f"init_adapter must be {expected}, got {tuple(init_adapter.shape)}")
                with torch.no_grad():
                    self.adapter.weight.copy_(init_adapter)

    @property
    def compute_dtype(self) -> torch.dtype:
        """dtype a chunk of hidden states must be cast to before scoring."""
        return self.weight.dtype if self.mode == "linear" else self.adapter.weight.dtype

    def refresh(self):
        """Drop the materialised weight; the next call rebuilds it."""
        self._cached_weight = None
        self._cache_key = None
        return self

    def materialized_weight(self) -> torch.Tensor:
        if self.mode == "linear":
            return self.weight
        weight = self.adapter.weight
        key = (weight._version, weight.data_ptr())
        if self._cached_weight is None or self._cache_key != key:
            # The backbone embedding is bfloat16 while the adapter is float32 (the
            # trainable dtype), so the product needs one common dtype. Compute it in the
            # adapter's dtype *with autocast disabled*: under accelerate's bf16 autocast a
            # float32 matmul would silently come back bfloat16 and then meet the float32
            # chunk in the loss. The bfloat16 -> float32 copy is a transient the caching
            # allocator reuses (1.2 GB for Qwen3-1.7B, once per step).
            embedding = self._embedding_getter()
            with no_autocast(weight.device):
                materialized = embedding.to(weight.dtype) @ weight
            if torch.is_grad_enabled() and weight.requires_grad:
                self._cached_weight = materialized
                self._cache_key = key
                return self._cached_weight
            # Under torch.no_grad() -- which is where every evaluation runs -- the product
            # carries no edge back to the adapter. Caching it under the *current* version
            # would make the next training forward score with a weight the adapter is not
            # part of, so the adapter would receive no gradient at all and DDP would abort
            # that step with "Expected to have finished reduction in the prior iteration
            # ... Parameter indices which did not receive grad: 757" (that index is this
            # adapter). Return the tensor without touching the cache instead: the next
            # grad-enabled forward materialises a connected weight again.
            return materialized
        return self._cached_weight

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        with no_autocast(hidden.device):
            return F.linear(hidden, self.materialized_weight())


class StaticLoRALinear(nn.Module):
    """PEFT-compatible static LoRA fallback used when ``peft`` is unavailable.

    The frozen base layer keeps the backbone dtype, while ``lora_A``/``lora_B`` may be
    float32 so that their AdamW state is not quantised to bfloat16. The two dtypes are
    reconciled at the module boundary: the LoRA branch computes in its own dtype with
    autocast disabled and the delta is cast back before it is added to the base output.
    This is the same pattern PEFT uses (``previous_dtype = x.dtype`` ... cast back).
    """
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0, dtype: torch.dtype | None = None):
        super().__init__(); self.base = base; self.rank = rank; self.scaling = alpha / rank
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features, dtype=dtype))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank, dtype=dtype))
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        for p in base.parameters(): p.requires_grad = False
    def forward(self, x):
        base_dtype = self.base.weight.dtype
        base_out = self.base(x.to(base_dtype))
        with no_autocast(x.device):
            lora_x = self.dropout(x).to(self.lora_A.dtype)
            delta = self.scaling * (lora_x @ self.lora_A.t() @ self.lora_B.t())
        return base_out + delta.to(base_dtype)


def disable_autocast_for_peft_lora(model: nn.Module) -> int:
    """Run every PEFT LoRA layer's forward with autocast disabled.

    ``StaticLoRALinear`` disables autocast at its own module boundary, which is what
    keeps its float32 ``lora_A``/``lora_B`` matmuls in float32. PEFT does the dtype
    casts itself but leaves autocast on, so under Accelerate's bf16 autocast its
    float32 LoRA weights are silently downcast again -- the precision the float32
    parameters exist for is lost.

    Wrapping the layer is safe for the frozen base matmul: its weights are already
    bf16, so with autocast off it runs in bf16, exactly as autocast would have run
    it. Only the float32 LoRA branch changes behaviour.

    Returns the number of wrapped modules (0 when no PEFT layer is present).
    """
    wrapped = 0
    for module in model.modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue
        if getattr(module, "_autocast_disabled", False):
            continue
        original = module.forward

        def forward_without_autocast(*args, __original=original, **kwargs):
            device = args[0].device if args and torch.is_tensor(args[0]) else "cpu"
            with no_autocast(device):
                return __original(*args, **kwargs)

        # Plain function, not a bound method: ``nn.Module.__call__`` reads
        # ``self.forward`` and invokes it with the caller's arguments only.
        module.forward = forward_without_autocast
        module._autocast_disabled = True
        wrapped += 1
    return wrapped


def build_block_causal_mask(
    context_mask: torch.Tensor,
    memory_length: int,
    question_mask: torch.Tensor,
    answer_mask: torch.Tensor,
    dtype: torch.dtype,
    slot_attention: str = "isolated",
    readout_length: int = 0,
) -> torch.Tensor:
    """Build an additive mask for ``[context, memory, question, readout, answer]``.

    Context positions attend only to valid, earlier context positions. Memory
    positions attend to valid context positions; slot-to-slot visibility is isolated,
    causal (including self), or bidirectional (including self). Question positions attend to memory and
    earlier question positions, read-out positions (``readout_length`` > 0) attend to
    memory and the question, and answer positions attend to memory, question, read-out and
    earlier answer positions. Padding keys are blocked. Padded query rows keep a self
    edge to avoid all-``-inf`` rows; their outputs are ignored by the loss.

    The memory rows are the one kind of row that can end up fully blocked: they see
    nothing but the context, so a row whose context is entirely padding leaves them with
    no key. This does **not** produce NaN, because blocked entries are written as
    ``torch.finfo(dtype).min`` -- a finite number -- so such a row's softmax is uniform
    (all-equal logits) rather than ``0/0``. Measured on the real Qwen3 attention with a
    fully blocked row: finite for both ``eager`` and ``sdpa``; substituting ``-inf`` for
    ``finfo.min`` does produce NaN under ``eager``, which is why the finite value matters.
    The uniformity is still meaningless, so ``src/data.py`` drops contexts that tokenize
    to nothing, making an all-padding context row unreachable in practice; the padding
    self-edge below is what keeps *context and QA* rows well defined regardless.

    This is the vectorised form of the original row-by-row builder. The loop version
    read one GPU scalar per row (``if c_valid[i]:``), which forces a device
    synchronisation and a separate kernel launch per row; at context length 2048
    that cost ~490 ms per call, more than a whole training step. Both forms produce
    identical masks.

    With an empty context this builder's question/answer rows are exactly
    :func:`build_continuation_mask`. The two are kept separate (deriving one from the
    other measured ~30% slower on the per-step hot path) and pinned together by
    ``tests/test_masks.py``.
    """
    if slot_attention not in ("isolated", "causal", "bidirectional"):
        raise ValueError("slot_attention must be isolated, causal, or bidirectional")
    context_mask = context_mask.bool(); question_mask = question_mask.bool(); answer_mask = answer_mask.bool()
    bsz, context_len = context_mask.shape
    question_len, answer_len = question_mask.shape[1], answer_mask.shape[1]
    readout_length = int(readout_length)
    if readout_length < 0:
        raise ValueError("readout_length must be >= 0")
    total = context_len + memory_length + question_len + readout_length + answer_len
    device = context_mask.device
    m0 = context_len
    q0 = context_len + memory_length
    r0 = q0 + question_len
    a0 = r0 + readout_length
    allowed = torch.zeros((bsz, total, total), dtype=torch.bool, device=device)
    # context rows: causal over valid context keys only
    causal_context = torch.tril(torch.ones(context_len, context_len, dtype=torch.bool, device=device))
    allowed[:, :context_len, :context_len] = context_mask[:, None, :] & causal_context
    # memory rows: every valid context key, nothing else (no memory <-> memory)
    allowed[:, m0:q0, :context_len] = context_mask[:, None, :]
    if slot_attention == "causal":
        # Let each memory token read the earlier memory tokens too, so the M slots can
        # coordinate instead of being independent bottleneck channels. ICAE's memory
        # tokens are ordinary causal positions and do see each other. Costs no memory
        # and no parameters; it only changes what the memory rows attend to.
        causal_memory = torch.tril(
            torch.ones(memory_length, memory_length, dtype=torch.bool, device=device)
        )
        allowed[:, m0:q0, m0:q0] = causal_memory
    elif slot_attention == "bidirectional":
        allowed[:, m0:q0, m0:q0] = True
    causal_qa = torch.tril(torch.ones(question_len + readout_length + answer_len, question_len + readout_length + answer_len, dtype=torch.bool, device=device))
    # question rows: all memory + causal over valid question keys
    allowed[:, q0:r0, m0:q0] = True
    allowed[:, q0:r0, q0:r0] = question_mask[:, None, :] & causal_qa[:question_len, :question_len]
    if readout_length:
        # read-out rows: all memory + every valid question key. They are derived from the
        # question and the memory, so they must not see the answer.
        allowed[:, r0:a0, m0:q0] = True
        allowed[:, r0:a0, q0:r0] = question_mask[:, None, :]
    # answer rows: all memory + all valid question keys (+ read-out) + causal over answers
    allowed[:, a0:, m0:q0] = True
    allowed[:, a0:, q0:r0] = question_mask[:, None, :]
    if readout_length:
        allowed[:, a0:, r0:a0] = True
    allowed[:, a0:, a0:] = answer_mask[:, None, :] & causal_qa[question_len + readout_length:, question_len + readout_length:]
    # padding query rows keep a single self edge
    context_rows = torch.arange(context_len, device=device)
    context_eye = torch.zeros((bsz, context_len, total), dtype=torch.bool, device=device)
    context_eye[:, context_rows, context_rows] = True
    allowed[:, :context_len] = torch.where(context_mask[:, :, None], allowed[:, :context_len], context_eye)
    qa_len = question_len + readout_length + answer_len
    qa_rows = torch.arange(qa_len, device=device)
    qa_eye = torch.zeros((bsz, qa_len, total), dtype=torch.bool, device=device)
    qa_eye[:, qa_rows, q0 + qa_rows] = True
    qa_valid = torch.cat(
        [question_mask, question_mask.new_ones(bsz, readout_length), answer_mask], dim=1
    )
    allowed[:, q0:] = torch.where(qa_valid[:, :, None], allowed[:, q0:], qa_eye)
    mask = torch.zeros((bsz, 1, total, total), dtype=dtype, device=device)
    return mask.masked_fill(~allowed[:, None], torch.finfo(dtype).min)


def build_continuation_mask(
    question_mask: torch.Tensor,
    answer_mask: torch.Tensor,
    memory_length: int,
    dtype: torch.dtype,
    readout_length: int = 0,
) -> torch.Tensor:
    """Mask for question/read-out/answer tokens attending to a memory-only cache.

    Vectorised form of the original row-by-row builder; both produce identical masks.
    With ``readout_length=0`` this is exactly :func:`build_block_causal_mask` with an empty
    context, sliced to drop the memory query rows -- ``tests/test_masks.py`` asserts that
    equivalence for both values so the two cannot drift apart.

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


class MetaLoRA(nn.Module):
    """Qwen encoder/decoder with ordinary (static) PEFT LoRA and recon decoders.

    Two dtypes coexist. The frozen Qwen backbone stays in its checkpoint dtype
    (``qwen_dtype``, bfloat16 for Qwen3) and every tensor that enters it is cast to
    that dtype. The trainable pieces -- ``memory_tokens``, ``lora_A``/``lora_B`` and
    the recon decoders -- can be kept in ``trainable_dtype`` (float32 by
    default) so that AdamW's moments and its ``param.add_(update, alpha=-lr)`` step
    are not quantised to bfloat16 resolution. The casts that reconcile the two dtypes
    live in three places: ``memory_tokens`` is cast down when it is used as an input,
    ``StaticLoRALinear``/``MemoryDecoder`` cast on entry and exit, and the losses
    upcast before reducing.
    """
    def __init__(self, qwen: nn.Module, rank=8, alpha=16.0, memory_length=8, decoder_hidden_size=256, decoder_heads=8, decoder_ffn_ratio=2, target_modules=None, dropout=0.0, max_context_tokens=2048, trainable_dtype=torch.float32, token_recon=False, embedding_recon=True, use_peft=False, head_mode="linear", head_init="auto", init_mode="randn", init_seed=0, slot_attention="isolated", ae_lm=False, readout_length=0, readout_layers=2, readout_heads=4, readout_hidden_size=256):
        super().__init__()
        self.rank, self.alpha = rank, alpha
        self.target_modules = tuple(("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj") if target_modules is None else target_modules)
        self.trainable_dtype = trainable_dtype
        # Which LoRA implementation is active, recorded so a run is reproducible.
        self.lora_backend = ("peft" if use_peft else "static") if self.target_modules else "none"
        # Only one of the two auxiliary objectives is active at a time.
        #   token_recon=True  -> decoder hidden states are classified into context tokens
        #   token_recon=False -> decoder outputs are regressed onto context input embeddings
        self.token_recon = bool(token_recon)
        self.embedding_recon = bool(embedding_recon)
        self.qwen = self._add_peft_lora(qwen, rank, alpha, dropout, bool(use_peft))
        self.qwen_hidden_size = self.qwen.config.hidden_size
        qwen_dtype = self.qwen.get_input_embeddings().weight.dtype
        # Global learnable memory tokens are added to the pooled Qwen context
        # representation. They are trained together with LoRA and the decoders.
        # Created in the trainable dtype; they are cast to the backbone dtype only
        # where they are used as an input, so gradients come back in float32.
        self.memory_tokens = nn.Parameter(
            torch.randn(memory_length, self.qwen_hidden_size, dtype=trainable_dtype) * 0.02
        )
        layers = int(self.qwen.config.num_hidden_layers)
        self.decoders = nn.ModuleList([
            MemoryDecoder(
                self.qwen_hidden_size, decoder_hidden_size, decoder_heads,
                decoder_ffn_ratio, max_context_tokens,
                reconstruct_embeddings=self.embedding_recon,
            )
            for _ in range(layers if self.embedding_recon or self.token_recon else 0)
        ])
        # Shared unembedding for the context next-token objective. One head is shared
        # by every layer decoder: a per-layer head would add
        # num_layers * D * vocab_size = 28 * 256 * 151936 ~= 1.09B parameters, while a
        # shared one costs D * vocab_size ~= 39M and still leaves each layer's decoder
        # free to produce its own hidden state.
        #
        # ``head_mode="tied"`` replaces those 39M from-scratch weights with a 0.5M
        # adapter into the backbone's own frozen embedding geometry (VocabularyHead).
        # ``head_init="auto"`` initialises that adapter from the decoders' memory
        # projection, so the first logits already mean "the token whose embedding the
        # memory points at" instead of a random direction.
        self.head_mode = head_mode
        self.token_recon_head = (
            VocabularyHead(
                decoder_hidden_size,
                int(self.qwen.get_input_embeddings().weight.size(0)),
                mode=head_mode,
                embedding_getter=lambda: self.qwen.get_input_embeddings().weight,
                backbone_hidden_size=self.qwen_hidden_size,
                init_adapter=self._head_adapter_init(head_init, decoder_hidden_size),
            )
            if self.token_recon else None
        )
        # Qwen's embedding weight is the single source of truth for the frozen
        # backbone dtype. It must not drag the trainable modules back to bfloat16,
        # so the trainable dtype is restored immediately afterwards.
        #
        # ``ae_lm`` additionally arms the memory-prefixed autoencoding pass, which reads
        # the context back out through the backbone's own (tied) unembedding. It carries
        # no parameters of its own.
        self.ae_lm = bool(ae_lm)
        self.ae_head = (
            TiedUnembedding(lambda: self.qwen.get_input_embeddings().weight)
            if self.ae_lm else None
        )
        # Per-question read-out over the (question-agnostic, cacheable) memory. Zero means
        # the historical behaviour: the answer attends only to the memory and the question.
        self.readout_length = int(readout_length)
        self.resampler = (
            QuestionResampler(
                self.qwen_hidden_size, self.readout_length, readout_layers, readout_heads,
                width=readout_hidden_size, dtype=trainable_dtype,
            )
            if self.readout_length > 0 else None
        )
        self.to(dtype=qwen_dtype)
        self.set_trainable_dtype(trainable_dtype)
        # Which memory-writer options are active, recorded for reproducibility. The
        # slot embeddings are (re)initialised last so the copy lands in the trainable
        # dtype and cannot be overwritten by the dtype casts above.
        self.init_mode = init_mode
        self.init_seed = int(init_seed)
        if slot_attention not in ("isolated", "causal", "bidirectional"):
            raise ValueError("slot_attention must be isolated, causal, or bidirectional")
        self.slot_attention = slot_attention
        self._init_memory_tokens(init_mode, init_seed)

    def _init_memory_tokens(self, init_mode, init_seed):
        """(Re)initialise the M slot embeddings. ``randn`` keeps the historical default.

        The scale of ``randn * 0.02`` already matches a token embedding, but the
        *directions* are a random subspace: the frozen backbone then sees memory
        positions that are off the manifold of real tokens. ``token_embed`` draws M
        distinct vocabulary entries instead, ``vocab_mean`` starts from the embedding
        mean with the same 0.02 noise. Both are writer-side engineering hygiene, not a
        mechanism from the compression papers.
        """
        if init_mode == "randn":
            return
        if init_mode not in {"token_embed", "vocab_mean"}:
            raise ValueError(
                f"memory.init_mode must be randn, token_embed or vocab_mean, got {init_mode!r}"
            )
        embedding = self.qwen.get_input_embeddings().weight
        with torch.no_grad():
            if init_mode == "token_embed":
                vocab = embedding.size(0)
                slots = self.memory_tokens.size(0)
                if slots > vocab:
                    raise ValueError(
                        f"memory_length {slots} exceeds the vocabulary size {vocab}"
                    )
                generator = torch.Generator().manual_seed(int(init_seed))
                ids = torch.randperm(vocab, generator=generator)[:slots]
                self.memory_tokens.data.copy_(embedding[ids].to(self.memory_tokens.dtype))
            else:
                mean = embedding.mean(dim=0).to(self.memory_tokens.dtype)
                noise = torch.randn(
                    self.memory_tokens.shape,
                    generator=torch.Generator().manual_seed(int(init_seed)),
                )
                self.memory_tokens.data.copy_(
                    mean + noise.to(self.memory_tokens.dtype) * 0.02
                )

    def set_trainable_dtype(self, dtype: torch.dtype):
        """Cast every trainable parameter to ``dtype`` (float32 by default).

        The backbone is untouched. Called automatically at the end of ``__init__`` so
        that the global ``self.to(dtype=qwen_dtype)`` cast does not flatten the
        trainable parameters back to the backbone dtype. Works for both the PEFT path
        and the ``StaticLoRALinear`` fallback because it matches on parameter names.
        """
        self.trainable_dtype = dtype
        for name, parameter in self.named_parameters():
            if is_trainable_parameter_name(name):
                parameter.data = parameter.data.to(dtype)
        return self

    def trainable_parameter_dtypes(self) -> dict[str, str]:
        dtypes: dict[str, str] = {}
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                dtypes[str(parameter.dtype)] = dtypes.get(str(parameter.dtype), 0) + 1
        return dtypes

    def _add_peft_lora(self, qwen, rank, alpha, dropout, use_peft):
        """Attach LoRA. Defaults to the static implementation, which is the audited one.

        The PEFT branch is opt-in (``model.use_peft: true``) because it is a
        different implementation of the same adapter: it needs
        :func:`disable_autocast_for_peft_lora` to honour the float32 contract, and
        the two paths are not covered by the same tests. Choosing it must be a
        deliberate act, not a side effect of whether ``peft`` happens to be
        installed -- ``peft`` is in the ``[train]`` extra, so following the README
        install command used to flip the implementation silently.
        """
        if not self.target_modules:
            for p in qwen.parameters():
                p.requires_grad = False
            return qwen
        if use_peft:
            try:
                from peft import LoraConfig, get_peft_model
            except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
                raise RuntimeError(
                    "model.use_peft=true but peft is not importable; install it "
                    "(pip install peft) or leave model.use_peft unset"
                ) from exc
            cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none", target_modules=list(self.target_modules), task_type="CAUSAL_LM")
            model = get_peft_model(qwen, cfg)
            for name, p in model.named_parameters():
                # Same rule as everywhere else: is_trainable_parameter_name is the single
                # source of truth, so adding a trainable module cannot miss this branch.
                p.requires_grad = is_trainable_parameter_name(name)
            disable_autocast_for_peft_lora(model)
            return model
        for p in qwen.parameters():
            p.requires_grad = False
        for name, module in list(qwen.named_modules()):
            if not isinstance(module, nn.Linear) or not any(name.endswith(x) for x in self.target_modules): continue
            parent=qwen
            for part in name.split('.')[:-1]: parent=getattr(parent, part)
            setattr(parent, name.split('.')[-1], StaticLoRALinear(module, rank, alpha, dropout, dtype=self.trainable_dtype))
        return qwen

    @property
    def dtype(self) -> torch.dtype:
        return self.qwen.get_input_embeddings().weight.dtype

    def _head_adapter_init(self, head_init, decoder_hidden_size):
        """Initial ``[H, D]`` adapter for a tied vocabulary head, or None for default.

        ``memory_projection`` uses the decoders' own memory projection. The adapter maps
        a decoder state back to the backbone width, so ``h @ W_mem`` is the up-projection
        of that state; scoring it against the frozen embedding gives "the tokens whose
        embeddings the memory already points at" at step 0. The head is shared by every
        layer, so the mean projection over layers is the symmetric choice.
        """
        if head_init == "random":
            return None
        if head_init == "auto":
            head_init = "memory_projection"
        if head_init != "memory_projection":
            raise ValueError(
                f"memory.head_init must be auto, random or memory_projection, got {head_init!r}"
            )
        with torch.no_grad():
            stacked = torch.stack(
                [decoder.memory_projection.weight for decoder in self.decoders], dim=0
            ).mean(dim=0)                       # [D, H]
        if tuple(stacked.shape) != (decoder_hidden_size, self.qwen_hidden_size):
            raise ValueError(
                f"memory projection has shape {tuple(stacked.shape)}, expected "
                f"({decoder_hidden_size}, {self.qwen_hidden_size})"
            )
        return stacked.t().detach().clone()     # [H, D]

    def _memory_prefix(self, layer_memory):
        return layer_memory[:, -1]  # final Qwen layer memory is the input prefix

    def _recon(self, layer_memory, context_embeds, context_mask):
        return torch.stack([decoder(layer_memory[:, i], context_embeds.size(1)) for i, decoder in enumerate(self.decoders)], dim=1)

    def sample_context_targets(self, context_ids, context_mask, positions_per_context=None):
        """Draw the context positions that the next-token objective will score.

        A context position ``t`` is a valid target when ``t >= 1`` (query ``t - 1`` must
        exist) and ``context_mask`` marks it as a real token. At most
        ``positions_per_context`` of them are kept per context, sampled without
        replacement; fewer are used for short contexts and the returned mask says
        which rows are real.

        Scoring all positions of a 2048-token context would apply the vocabulary head
        to ``num_layers * 2048`` rows, which costs about as much as the whole Qwen
        forward pass. Sampling a few hundred positions keeps the auxiliary objective
        under ~10% overhead while still covering the context uniformly.
        """
        if context_ids.ndim != 2 or context_mask.shape != context_ids.shape:
            raise ValueError("context_ids and context_mask must both be [B, L]")
        device = context_ids.device
        length = context_ids.size(1)
        candidates = context_mask.bool().clone()
        if length:
            candidates[:, 0] = False          # no target before the first token
        budget = length - 1 if not positions_per_context or positions_per_context <= 0 else int(positions_per_context)
        budget = max(1, min(budget, max(1, length - 1)))
        scores = torch.rand(
            context_ids.size(0), length, device=device,
        ).masked_fill(~candidates, float("-inf"))
        positions = scores.topk(budget, dim=1).indices
        keep = candidates.gather(1, positions)
        positions = positions.masked_fill(~keep, 0)
        return positions, keep

    def token_recon_terms(self, context_ids, context_mask, layer_memory, positions_per_context=None):
        """Build the context next-token prediction terms from the memory.

        The query for target position ``t`` sits at ``t - 1``, so the decoder has to
        produce token ``t`` from the memory and its own positional query alone -- it
        never sees token ``t`` or any other context token. That is what forces context
        content into the memory instead of into the decoder.
        """
        if layer_memory is None:
            raise ValueError("token_recon requires the prefix layer_memory")
        positions, keep = self.sample_context_targets(context_ids, context_mask, positions_per_context)
        query_positions = (positions - 1).clamp_min(0)
        labels = context_ids.gather(1, positions).masked_fill(~keep, -100)
        hidden = torch.stack(
            [decoder.decode(layer_memory[:, i], query_positions) for i, decoder in enumerate(self.decoders)],
            dim=1,
        )
        return ContextLMTerms(hidden, labels, keep)

    @property
    def _transformer_body(self):
        """The transformer without the LM head.

        ``encode_context_prefix`` only reads ``hidden_states`` and ``past_key_values``,
        but ``load_model`` builds a ``Qwen3ForCausalLM``, so calling ``self.qwen`` would
        additionally project every context position through the 2048 x 151936 vocabulary
        head: at context length 2048 that is a wasted matmul plus a ~0.6 GiB logits tensor
        per row. When PEFT is installed the transformer sits one wrapper deeper, so peel
        off anything that exposes an ``lm_head``.
        """
        module = self.qwen
        for _ in range(4):
            if not hasattr(module, "lm_head"):
                return module
            inner = getattr(module, "base_model", None) or getattr(module, "model", None)
            if inner is None:
                break
            module = inner
        return self.qwen

    def encode_context_prefix(self, context_embeds, context_mask) -> ContextPrefix:
        """Encode each context and memory once, retaining a differentiable KV cache."""
        context_embeds = context_embeds.to(dtype=self.dtype)
        context_mask = context_mask.bool()
        # Cast down to the backbone dtype only here, where the tokens become an
        # input to Qwen; the parameter itself stays in the trainable dtype.
        memory_inputs = self.memory_tokens.to(dtype=self.dtype).unsqueeze(0).expand(context_embeds.size(0), -1, -1)
        sequence = torch.cat([context_embeds, memory_inputs], dim=1)
        empty = context_mask.new_zeros(context_embeds.size(0), 0)
        block_mask = build_block_causal_mask(context_mask, memory_inputs.size(1), empty, empty, sequence.dtype, slot_attention=self.slot_attention)
        # No logits are read from this pass, so skip the vocabulary head entirely.
        out = self._transformer_body(inputs_embeds=sequence, attention_mask=block_mask, output_hidden_states=True, use_cache=True, return_dict=True)
        states = out.hidden_states or (sequence, out.last_hidden_state)
        layer_memory = []
        for state in states[1:1 + int(self.qwen.config.num_hidden_layers)]:
            layer_memory.append(state[:, context_embeds.size(1):context_embeds.size(1) + memory_inputs.size(1)])
        while len(layer_memory) < int(self.qwen.config.num_hidden_layers):
            layer_memory.append(layer_memory[-1])
        layer_memory = torch.stack(layer_memory, dim=1)
        cache = out.past_key_values
        if cache is None:
            raise RuntimeError("Qwen did not return a KV cache for prefix encoding")
        memory_layers = []
        for layer in cache.layers:
            # DynamicCache uses [batch, kv_heads, sequence, head_dim].
            keys = layer.keys[..., context_embeds.size(1):, :]
            values = layer.values[..., context_embeds.size(1):, :]
            memory_layers.append((keys, values))
        memory_cache = DynamicCache(ddp_cache_data=memory_layers, config=self.qwen.config)
        recon = self._recon(layer_memory, context_embeds, context_mask) if self.embedding_recon else None
        return ContextPrefix(layer_memory, self._memory_prefix(layer_memory), memory_cache, recon, context_embeds, context_mask, context_embeds.size(1), memory_inputs.size(1), context_hidden=states[-1][:, :context_embeds.size(1)])

    @staticmethod
    def _select_cache(cache, indices, config):
        layers = []
        for layer in cache.layers:
            layers.append((layer.keys.index_select(0, indices), layer.values.index_select(0, indices)))
        return DynamicCache(ddp_cache_data=layers, config=config)

    def _readout_embeddings(self, prefix, question_embeds, question_mask, indices=None):
        """``[B, R, H]`` question-conditioned positions, or None when the read-out is off.

        ``indices`` selects which memory rows the questions belong to (the training collate
        flattens several QA rows per context, and generation groups rows by context), so the
        resampled memory always matches the question row it is paired with.
        """
        if self.resampler is None:
            return None
        if prefix.memory is None:
            raise RuntimeError("the question read-out needs the memory representation")
        memory = prefix.memory if indices is None else prefix.memory.index_select(0, indices)
        readout = self.resampler(question_embeds, question_mask, memory)
        return readout.to(dtype=self.dtype)

    def _qa_positions(self, prefix, question_mask, answer_length, readout_length, device):
        """Absolute positions of ``[question, (read-out), answer]`` behind a memory prefix.

        The numbering is the one the generation path builds row by row
        (:meth:`generate_answers_with_prefix`), and it deliberately ignores left padding::

            start = context_length + memory_length
            real question token i (0-based among the real ones) -> start + i
            read-out slot j                                     -> start + valid + j
            answer token k                                      -> start + valid + readout_length + k

        It used to be a single ``arange`` over the padded block, so a row's real question
        tokens sat at ``start + pad + i`` and its answer at ``start + padded_question +
        readout``. That made the memory-to-question distance depend on the longest question
        in the batch, and generation -- which compacts the question before decoding -- used
        a different numbering than training. Both are the same now; with no padding in the
        batch the result is identical to the old ``arange``, which is why the old form was
        right for single-row batches.
        """
        question_mask = question_mask.bool()
        valid = question_mask.sum(dim=1)
        start = int(prefix.context_length) + int(prefix.memory_length)
        # cumsum-1 is the 0-based rank of each real token; padding slots get -1 and are
        # parked at ``start`` (they are masked out of attention either way).
        rank = question_mask.long().cumsum(dim=1) - 1
        blocks = [torch.where(question_mask, start + rank, torch.full_like(rank, start))]
        if readout_length:
            blocks.append(
                start + valid.unsqueeze(1)
                + torch.arange(readout_length, device=device).unsqueeze(0)
            )
        if answer_length:
            blocks.append(
                start + valid.unsqueeze(1) + readout_length
                + torch.arange(answer_length, device=device).unsqueeze(0)
            )
        return torch.cat(blocks, dim=1).to(device)

    def forward_qa_with_prefix(self, prefix, qa_context_indices, question_embeds, question_mask, answer_embeds, answer_mask, labels):
        question_embeds, answer_embeds = question_embeds.to(dtype=self.dtype), answer_embeds.to(dtype=self.dtype)
        indices = qa_context_indices.to(question_embeds.device, dtype=torch.long)
        cache = self._select_cache(prefix.memory_cache, indices, self.qwen.config)
        readout = self._readout_embeddings(prefix, question_embeds, question_mask, indices)
        extra = 0 if readout is None else readout.size(1)
        if readout is None:
            sequence = torch.cat([question_embeds, answer_embeds], dim=1)
        else:
            sequence = torch.cat([question_embeds, readout, answer_embeds], dim=1)
        mask = build_continuation_mask(question_mask, answer_mask, prefix.memory_length, sequence.dtype, readout_length=extra)
        positions = self._qa_positions(prefix, question_mask, answer_embeds.size(1), extra, sequence.device)
        out = self.qwen(inputs_embeds=sequence, attention_mask=mask, position_ids=positions, past_key_values=cache, use_cache=True, return_dict=True)
        ignored = torch.full((labels.size(0), question_embeds.size(1) + extra), -100, dtype=labels.dtype, device=labels.device)
        return MetaLoRAOutput(
            logits=out.logits,
            labels=torch.cat([ignored, labels], dim=1),
            memory=prefix.memory,
            layer_memory=prefix.layer_memory,
            recon=prefix.recon,
            context_target=prefix.context_target,
            context_mask=prefix.context_mask,
        )

    def autoencode_with_memory(self, prefix, context_embeds, context_mask):
        """Teacher-forced recon of the context through the memory prefix.

        This is the autoencoding objective the context-compression literature uses (e.g.
        500xCompressor eq. 1): the *frozen backbone* is the decoder, the per-layer key/value
        pairs of the memory prefix it, and every context token is predicted from the memory
        plus the teacher-forced prefix, ``P(t_i | memory, t_<i)``. The memory slots and the LoRA
        adapters still receive gradient because the prefix is differentiable, but no
        decoder has to be learned from scratch -- which is exactly what separates this from
        the memory-only ``token_recon`` objective (where every token must be produced from
        the memory alone).

        The continuation mask is reused: its "question rows" rule is "read all memory, then
        attend causally to yourself", which is precisely teacher forcing over the context.
        Positions start after the memory, matching ``forward_qa_with_prefix``, so the memory
        keys keep the positions they were encoded with and no RoPE surgery is needed.

        Returns the last hidden state ``[B, L, H]``. The caller scores it with the
        backbone's own (tied) unembedding through :func:`src.losses.sequence_lm_loss`, so
        vocabulary-sized logits are never materialised for all positions at once.
        """
        context_embeds = context_embeds.to(dtype=self.dtype)
        context_mask = context_mask.bool()
        empty = context_mask.new_zeros(context_mask.size(0), 0)
        mask = build_continuation_mask(context_mask, empty, prefix.memory_length, context_embeds.dtype)
        start = prefix.context_length + prefix.memory_length
        positions = torch.arange(start, start + context_embeds.size(1), device=context_embeds.device)
        positions = positions.unsqueeze(0).expand(context_embeds.size(0), -1)
        # Qwen updates a supplied DynamicCache even with use_cache=False. Keep
        # reconstruction's context tokens out of the reusable memory-only prefix.
        indices = torch.arange(context_embeds.size(0), device=context_embeds.device)
        cache = self._select_cache(prefix.memory_cache, indices, self.qwen.config)
        out = self._transformer_body(
            inputs_embeds=context_embeds,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=cache,
            use_cache=False,
            return_dict=True,
        )
        return out.last_hidden_state

    def forward(self, context_embeds, context_mask, question_embeds, question_mask, answer_embeds, answer_mask, labels, qa_context_indices=None, context_ids=None, token_recon_positions=None, objective_config=None):
        """Training path.

        ``context_ids`` is required when the auxiliary objective is ``token_recon``: the
        regression target could be read back out of ``context_embeds`` but a token
        classification target cannot, because the embedding lookup is not invertible. It is
        also required for the autoencoding objective (``ae_lm``).
        """
        prefix = self.encode_context_prefix(context_embeds, context_mask)
        indices = qa_context_indices if qa_context_indices is not None else torch.arange(question_embeds.size(0), device=question_embeds.device)
        output = self.forward_qa_with_prefix(prefix, indices, question_embeds, question_mask, answer_embeds, answer_mask, labels)
        if objective_config is not None:
            from .losses import memory_objectives
            output.objective_terms = memory_objectives(self, prefix, context_ids, context_mask, objective_config)
            return output
        if self.token_recon:
            if context_ids is None:
                raise ValueError("token_recon needs context_ids, but forward() received none")
            terms = self.token_recon_terms(
                context_ids.to(context_embeds.device), context_mask.to(context_embeds.device),
                prefix.layer_memory, token_recon_positions,
            )
            output.token_recon_hidden = terms.hidden
            output.token_recon_labels = terms.labels
            output.token_recon_mask = terms.mask
        if self.ae_lm:
            if context_ids is None:
                raise ValueError("the autoencoding objective needs context_ids, but forward() received none")
            context_ids = context_ids.to(context_embeds.device)
            context_mask = context_mask.to(context_embeds.device).bool()
            hidden = self.autoencode_with_memory(prefix, context_embeds, context_mask)
            # Align prediction and target here, once: hidden[i] scores context_ids[i + 1].
            # The first context token has no predecessor to be predicted from, so it is the
            # only unscored position.
            output.ae_hidden = hidden[:, :-1]
            output.ae_labels = context_ids[:, 1:]
            output.ae_mask = context_mask[:, 1:]
            # The encoder pass already computed the plain causal-LM states for the same
            # positions (its context rows attend only to earlier context), so the
            # distillation target costs nothing extra. Sliced the same way as ae_hidden.
            if prefix.context_hidden is not None:
                output.ae_teacher_hidden = prefix.context_hidden[:, :-1]
        return output

    @torch.no_grad()
    def generate_answers_with_prefix(self, prefix, qa_context_indices, question_ids, question_mask,
                                     tokenizer, max_new_tokens=128, group_by_context=True,
                                     max_rows_per_group=4):
        """Greedy decoding for many QA rows that share one context prefix.

        Rows belonging to the same context share the same memory KV cache, so they are
        prefilled and decoded as one batch instead of one row at a time. Each row keeps
        its own position ids, so a row's real question tokens occupy exactly the same
        absolute positions they would have in the row-by-row path; question padding is
        left of them and masked out. The batched result is therefore identical to
        decoding every row on its own, only faster.

        ``group_by_context=False`` decodes every row as its own group.
        """
        embedding = self.qwen.get_input_embeddings()
        device = question_ids.device
        start = prefix.context_length + prefix.memory_length
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

        per_row_tokens: list[list[int]] = [[] for _ in range(n_rows)]
        for group in groups:
            for chunk_start in range(0, len(group), max_rows_per_group):
                rows = group[chunk_start:chunk_start + max_rows_per_group]
                count = len(rows)
                row_index = torch.tensor(rows, dtype=torch.long, device=device)
                row_ids = question_ids[row_index]
                row_valid = question_mask[row_index].bool()
                valid = row_valid.sum(dim=1)                              # [count]
                width = max(1, int(valid.max()))
                padding = width - valid                                   # [count]
                # left-align the real question tokens, then restore each row's own
                # absolute positions so the layout matches the row-by-row decode
                q_ids = torch.full((count, width), int(tokenizer.pad_token_id), dtype=row_ids.dtype, device=device)
                q_mask = torch.zeros((count, width), dtype=torch.bool, device=device)
                for slot, row in enumerate(rows):
                    real = row_ids[slot][row_valid[slot]]
                    q_ids[slot, width - real.numel():] = real
                    q_mask[slot, width - real.numel():] = True
                columns = torch.arange(width, device=device).unsqueeze(0).expand(count, -1)
                positions = start + (columns - padding[:, None])
                positions = torch.where(q_mask, positions, torch.full_like(positions, start)).clamp_(min=0)

                cache = self._select_cache(
                    prefix.memory_cache,
                    qa_context_indices[row_index].to(device, dtype=torch.long),
                    self.qwen.config,
                )
                question = embedding(q_ids).to(dtype=self.dtype)
                readout = self._readout_embeddings(
                    prefix, question, q_mask,
                    qa_context_indices[row_index].to(device, dtype=torch.long),
                )
                extra = 0 if readout is None else readout.size(1)
                if readout is None:
                    sequence, sequence_positions = question, positions
                else:
                    # The read-out sits right after each row's real question tokens, exactly
                    # where a row-by-row decode would put it.
                    readout_positions = start + valid.unsqueeze(1) + torch.arange(extra, device=device).unsqueeze(0)
                    sequence = torch.cat([question, readout], dim=1)
                    sequence_positions = torch.cat([positions, readout_positions], dim=1)
                # Evaluation may run outside Accelerate's autocast context.  Keep the
                # complete embedding sequence in the frozen backbone dtype before it
                # enters Qwen/LoRA; otherwise a float32 readout or adapter branch can
                # reach a bf16 linear layer and fail with ``mat1 and mat2`` dtype errors.
                sequence = sequence.to(dtype=self.dtype)
                empty = q_mask.new_zeros(count, 0)
                mask = build_continuation_mask(q_mask, empty, prefix.memory_length, question.dtype, readout_length=extra)
                output = self.qwen(inputs_embeds=sequence, attention_mask=mask, position_ids=sequence_positions,
                                   past_key_values=cache, use_cache=True, return_dict=True)
                cache = output.past_key_values
                next_id = output.logits[:, -1].argmax(dim=-1)
                finished = torch.zeros(count, dtype=torch.bool, device=device)
                # Question padding must stay blocked for the incremental steps too: the
                # cache layout is [memory, question (with padding), read-out, generated...],
                # and a plain all-ones mask would expose the padding keys that the prefill
                # masked out.  Rows without padding make this a no-op.
                cache_prefix_mask = torch.cat(
                    [q_mask.new_ones(count, prefix.memory_length), q_mask, q_mask.new_ones(count, extra)], dim=1
                )
                for step in range(max_new_tokens):
                    for slot, row in enumerate(rows):
                        if not finished[slot]:
                            per_row_tokens[row].append(int(next_id[slot]))
                    if eos_id is not None:
                        finished |= next_id.eq(int(eos_id))
                    if bool(finished.all()) or step + 1 == max_new_tokens:
                        break
                    # Once a row is done its remaining steps are harmless: the tokens are
                    # no longer recorded and the loop stops when every row has finished.
                    token = embedding(next_id[:, None]).to(dtype=self.dtype)
                    attention_mask = torch.cat(
                        [cache_prefix_mask,
                         q_mask.new_ones(count, step + 1)],
                        dim=1,
                    )
                    position_ids = (start + valid + extra + step).unsqueeze(1)
                    output = self.qwen(inputs_embeds=token, attention_mask=attention_mask,
                                       position_ids=position_ids, past_key_values=cache,
                                       use_cache=True, return_dict=True)
                    cache = output.past_key_values
                    next_id = output.logits[:, -1].argmax(dim=-1)

        width_out = max((len(tokens) for tokens in per_row_tokens), default=0)
        result = torch.full((n_rows, width_out), tokenizer.pad_token_id, dtype=torch.long, device=device)
        for row, tokens in enumerate(per_row_tokens):
            if tokens:
                result[row, :len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
        return result

    @torch.no_grad()
    def generate_answer_with_prefix(self, prefix, qa_context_indices, question_ids, question_mask, tokenizer, max_new_tokens=128):
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


def load_model(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from utils.config import dtype_from_name
    tok=AutoTokenizer.from_pretrained(cfg.model.name_or_path, local_files_only=True); tok.pad_token=tok.pad_token or tok.eos_token
    qwen=AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path, dtype=dtype_from_name(cfg.model.torch_dtype), local_files_only=True)
    qwen.config.use_cache=False
    model=MetaLoRA(
        qwen, rank=cfg.model.lora_rank, alpha=cfg.model.lora_alpha,
        memory_length=cfg.memory.memory_length,
        decoder_hidden_size=cfg.memory.decoder_hidden_size,
        decoder_heads=cfg.memory.decoder_heads,
        decoder_ffn_ratio=cfg.memory.decoder_ffn_ratio,
        target_modules=cfg.model.target_modules, dropout=cfg.model.lora_dropout,
        max_context_tokens=cfg.data.max_context_tokens,
        trainable_dtype=dtype_from_name(cfg.model.trainable_dtype),
        token_recon=cfg.memory.token_recon_weight > 0,
        embedding_recon=cfg.memory.embedding_recon_weight > 0,
        use_peft=cfg.model.use_peft,
        head_mode=cfg.memory.head_mode,
        head_init=cfg.memory.head_init,
        init_mode=cfg.memory.init_mode,
        init_seed=cfg.memory.init_seed,
        slot_attention=cfg.memory.slot_attention,
        ae_lm=cfg.memory.causal_recon_weight > 0 or cfg.memory.distill_weight > 0,
        readout_length=cfg.memory.readout_length,
        readout_layers=cfg.memory.readout_layers,
        readout_heads=cfg.memory.readout_heads,
        readout_hidden_size=cfg.memory.readout_hidden_size,
    )
    for name,p in model.named_parameters():
        p.requires_grad=is_trainable_parameter_name(name)
    return tok, model


__all__ = [
    "MetaLoRA", "MetaLoRAOutput",
    "StaticLoRALinear", "MemoryDecoder", "ContextPrefix", "ContextLMTerms",
    "VocabularyHead",
    "TiedUnembedding",
    "QuestionResampler",
    "build_block_causal_mask", "build_continuation_mask",
    "is_trainable_parameter_name", "load_model",
    "disable_autocast_for_peft_lora",
]
