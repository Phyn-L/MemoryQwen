from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from .MemoryDecoder import MemoryDecoder

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
    reconstruction: torch.Tensor
    context_target: torch.Tensor
    context_mask: torch.Tensor


@dataclass
class ContextPrefix:
    layer_memory: torch.Tensor
    memory: torch.Tensor
    memory_cache: object
    reconstruction: torch.Tensor
    context_target: torch.Tensor
    context_mask: torch.Tensor
    context_length: int
    memory_length: int


class StaticLoRALinear(nn.Module):
    """PEFT-compatible static LoRA fallback used when ``peft`` is unavailable."""
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__(); self.base = base; self.rank = rank; self.scaling = alpha / rank
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        for p in base.parameters(): p.requires_grad = False
    def forward(self, x):
        base_dtype = self.base.weight.dtype
        base_out = self.base(x.to(base_dtype))
        lora_x = self.dropout(x.to(base_dtype))
        delta = self.scaling * (lora_x @ self.lora_A.t() @ self.lora_B.t())
        return base_out + delta


class LayerReconstructionDecoder(nn.Module):
    """Decode one Qwen layer's memory slots into every original context embedding."""
    def __init__(self, dim: int, heads: int, max_positions: int = 2048):
        super().__init__()
        self.position = nn.Embedding(max_positions, dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
    def forward(self, memory: torch.Tensor, length: int) -> torch.Tensor:
        q = self.position(torch.arange(length, device=memory.device))[None].expand(memory.size(0), -1, -1)
        h, _ = self.attn(q, memory, memory, need_weights=False)
        return self.norm(q + h + self.ff(q + h))


def build_block_causal_mask(
    context_mask: torch.Tensor,
    memory_length: int,
    question_mask: torch.Tensor,
    answer_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build an additive mask for ``[context, memory, question, answer]``.

    Context positions attend only to valid, earlier context positions. Memory
    positions attend only to valid context positions (never to another memory
    token). Question positions attend to memory and earlier question positions,
    while answer positions attend to memory, question, and earlier answer
    positions. Padding keys are blocked. Padded query rows keep a self edge to
    avoid all-``-inf`` rows; their outputs are ignored by the loss.
    """
    context_mask = context_mask.bool(); question_mask = question_mask.bool(); answer_mask = answer_mask.bool()
    bsz, context_len = context_mask.shape
    question_len, answer_len = question_mask.shape[1], answer_mask.shape[1]
    total = context_len + memory_length + question_len + answer_len
    allowed = torch.zeros((bsz, total, total), dtype=torch.bool, device=context_mask.device)
    c0, m0, q0 = 0, context_len, context_len + memory_length
    a0 = q0 + question_len
    for batch in range(bsz):
        c_valid = context_mask[batch]
        q_valid = question_mask[batch]
        a_valid = answer_mask[batch]
        for i in range(context_len):
            if c_valid[i]:
                allowed[batch, c0 + i, c0 : c0 + i + 1] = c_valid[: i + 1]
            else:
                allowed[batch, c0 + i, c0 + i] = True
        valid_memory = torch.ones(memory_length, dtype=torch.bool, device=allowed.device)
        for i in range(memory_length):
            allowed[batch, m0 + i, c0 : c0 + context_len] = c_valid
        for i in range(question_len):
            if q_valid[i]:
                allowed[batch, q0 + i, m0 : m0 + memory_length] = valid_memory
                allowed[batch, q0 + i, q0 : q0 + i + 1] = q_valid[: i + 1]
            else:
                allowed[batch, q0 + i, q0 + i] = True
        for i in range(answer_len):
            if a_valid[i]:
                allowed[batch, a0 + i, m0 : m0 + memory_length] = valid_memory
                allowed[batch, a0 + i, q0 : q0 + question_len] = q_valid
                allowed[batch, a0 + i, a0 : a0 + i + 1] = a_valid[: i + 1]
            else:
                allowed[batch, a0 + i, a0 + i] = True
    mask = torch.zeros((bsz, 1, total, total), dtype=dtype, device=allowed.device)
    return mask.masked_fill(~allowed[:, None], torch.finfo(dtype).min)


def build_continuation_mask(
    question_mask: torch.Tensor,
    answer_mask: torch.Tensor,
    memory_length: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Mask for question/answer tokens attending to a memory-only cache."""
    question_mask, answer_mask = question_mask.bool(), answer_mask.bool()
    bsz, question_length = question_mask.shape
    answer_length = answer_mask.shape[1]
    current = question_length + answer_length
    allowed = torch.zeros((bsz, current, memory_length + current), dtype=torch.bool, device=question_mask.device)
    for batch in range(bsz):
        q_valid, a_valid = question_mask[batch], answer_mask[batch]
        for i in range(question_length):
            if q_valid[i]:
                allowed[batch, i, :memory_length] = True
                allowed[batch, i, memory_length:memory_length + i + 1] = q_valid[: i + 1]
            else:
                allowed[batch, i, memory_length + i] = True
        for i in range(answer_length):
            row = question_length + i
            if a_valid[i]:
                allowed[batch, row, :memory_length] = True
                allowed[batch, row, memory_length:memory_length + question_length] = q_valid
                allowed[batch, row, memory_length + question_length:memory_length + question_length + i + 1] = a_valid[: i + 1]
            else:
                allowed[batch, row, memory_length + row] = True
    mask = torch.zeros((bsz, 1, current, memory_length + current), dtype=dtype, device=question_mask.device)
    return mask.masked_fill(~allowed[:, None], torch.finfo(dtype).min)


class MetaLoRA(nn.Module):
    """Qwen encoder/decoder with ordinary (static) PEFT LoRA and reconstruction decoders."""
    def __init__(self, qwen: nn.Module, rank=8, alpha=16.0, memory_length=8, decoder_hidden_size=256, decoder_heads=8, decoder_ffn_ratio=2, target_modules=None, dropout=0.0, max_context_tokens=2048):
        super().__init__()
        self.rank, self.alpha = rank, alpha
        self.target_modules = tuple(target_modules or ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))
        self.qwen = self._add_peft_lora(qwen, rank, alpha, dropout)
        self.qwen_hidden_size = self.qwen.config.hidden_size
        qwen_dtype = self.qwen.get_input_embeddings().weight.dtype
        # Global learnable memory tokens are added to the pooled Qwen context
        # representation. They are trained together with LoRA and the decoders.
        self.memory_tokens = nn.Parameter(
            torch.randn(memory_length, self.qwen_hidden_size, dtype=qwen_dtype) * 0.02
        )
        layers = int(self.qwen.config.num_hidden_layers)
        self.decoders = nn.ModuleList([
            MemoryDecoder(
                self.qwen_hidden_size, decoder_hidden_size, decoder_heads,
                decoder_ffn_ratio, max_context_tokens,
            )
            for _ in range(layers)
        ])
        # Qwen's embedding weight is the single source of truth for every
        # floating-point parameter and buffer in this composite model.
        self.to(dtype=qwen_dtype)

    def _add_peft_lora(self, qwen, rank, alpha, dropout):
        try:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none", target_modules=list(self.target_modules), task_type="CAUSAL_LM")
            model = get_peft_model(qwen, cfg)
            for name, p in model.named_parameters():
                p.requires_grad = ("lora_A" in name or "lora_B" in name)
            return model
        except (ImportError, ModuleNotFoundError):
            for p in qwen.parameters():
                p.requires_grad = False
            for name, module in list(qwen.named_modules()):
                if not isinstance(module, nn.Linear) or not any(name.endswith(x) for x in self.target_modules): continue
                parent=qwen
                for part in name.split('.')[:-1]: parent=getattr(parent, part)
                setattr(parent, name.split('.')[-1], StaticLoRALinear(module, rank, alpha, dropout))
            return qwen

    @property
    def base_model(self):
        return self.qwen

    @property
    def dtype(self) -> torch.dtype:
        return self.qwen.get_input_embeddings().weight.dtype

    def _encode_context(self, context_embeds, context_mask, question_embeds=None, question_mask=None, answer_embeds=None, answer_mask=None):
        context_embeds = context_embeds.to(dtype=self.dtype)
        if question_embeds is None:
            question_embeds = context_embeds.new_empty(context_embeds.size(0), 0, context_embeds.size(-1))
            question_mask = context_mask.new_zeros(context_embeds.size(0), 0)
        else:
            question_embeds = question_embeds.to(dtype=self.dtype)
        if answer_embeds is None:
            answer_embeds = context_embeds.new_empty(context_embeds.size(0), 0, context_embeds.size(-1))
            answer_mask = context_mask.new_zeros(context_embeds.size(0), 0)
        else:
            answer_embeds = answer_embeds.to(dtype=self.dtype)
        memory_length = self.memory_tokens.size(0)
        memory_inputs = self.memory_tokens.unsqueeze(0).expand(context_embeds.size(0), -1, -1)
        sequence = torch.cat([context_embeds, memory_inputs, question_embeds, answer_embeds], dim=1)
        block_mask = build_block_causal_mask(context_mask, memory_length, question_mask, answer_mask, sequence.dtype)
        out = self.qwen(inputs_embeds=sequence, attention_mask=block_mask, output_hidden_states=True, use_cache=False, return_dict=True)
        states = out.hidden_states
        if states is None: states = (sequence, out.last_hidden_state)
        per_layer=[]
        for state in states[1:1 + len(self.decoders)]:
            per_layer.append(state[:, context_embeds.size(1) : context_embeds.size(1) + memory_inputs.size(1)])
        while len(per_layer) < len(self.decoders): per_layer.append(per_layer[-1])
        return torch.stack(per_layer, dim=1), out, sequence, block_mask

    def _memory_prefix(self, layer_memory):
        return layer_memory[:, -1]  # final Qwen layer memory is the input prefix

    def _reconstruction(self, layer_memory, context_embeds, context_mask):
        return torch.stack([decoder(layer_memory[:, i], context_embeds.size(1)) for i, decoder in enumerate(self.decoders)], dim=1)

    def encode_context_prefix(self, context_embeds, context_mask) -> ContextPrefix:
        """Encode each context and memory once, retaining a differentiable KV cache."""
        context_embeds = context_embeds.to(dtype=self.dtype)
        context_mask = context_mask.bool()
        memory_inputs = self.memory_tokens.unsqueeze(0).expand(context_embeds.size(0), -1, -1)
        sequence = torch.cat([context_embeds, memory_inputs], dim=1)
        empty = context_mask.new_zeros(context_embeds.size(0), 0)
        block_mask = build_block_causal_mask(context_mask, memory_inputs.size(1), empty, empty, sequence.dtype)
        out = self.qwen(inputs_embeds=sequence, attention_mask=block_mask, output_hidden_states=True, use_cache=True, return_dict=True)
        states = out.hidden_states or (sequence, out.last_hidden_state)
        layer_memory = []
        for state in states[1:1 + len(self.decoders)]:
            layer_memory.append(state[:, context_embeds.size(1):context_embeds.size(1) + memory_inputs.size(1)])
        while len(layer_memory) < len(self.decoders):
            layer_memory.append(layer_memory[-1])
        layer_memory = torch.stack(layer_memory, dim=1)
        cache = out.past_key_values
        if cache is None:
            raise RuntimeError("Qwen did not return a KV cache for prefix encoding")
        memory_layers = []
        for layer in cache.layers:
            keys = layer.keys[..., context_embeds.size(1):, :]
            values = layer.values[..., context_embeds.size(1):, :]
            memory_layers.append((keys, values))
        memory_cache = DynamicCache(ddp_cache_data=memory_layers, config=self.qwen.config)
        reconstruction = self._reconstruction(layer_memory, context_embeds, context_mask)
        return ContextPrefix(layer_memory, self._memory_prefix(layer_memory), memory_cache, reconstruction, context_embeds, context_mask, context_embeds.size(1), memory_inputs.size(1))

    @staticmethod
    def _select_cache(cache, indices, config):
        layers = []
        for layer in cache.layers:
            layers.append((layer.keys.index_select(0, indices), layer.values.index_select(0, indices)))
        return DynamicCache(ddp_cache_data=layers, config=config)

    def forward_qa_with_prefix(self, prefix, qa_context_indices, question_embeds, question_mask, answer_embeds, answer_mask, labels):
        question_embeds, answer_embeds = question_embeds.to(dtype=self.dtype), answer_embeds.to(dtype=self.dtype)
        indices = qa_context_indices.to(question_embeds.device, dtype=torch.long)
        cache = self._select_cache(prefix.memory_cache, indices, self.qwen.config)
        sequence = torch.cat([question_embeds, answer_embeds], dim=1)
        mask = build_continuation_mask(question_mask, answer_mask, prefix.memory_length, sequence.dtype)
        start = prefix.context_length + prefix.memory_length
        positions = torch.arange(start, start + sequence.size(1), device=sequence.device).unsqueeze(0).expand(sequence.size(0), -1)
        out = self.qwen(inputs_embeds=sequence, attention_mask=mask, position_ids=positions, past_key_values=cache, use_cache=True, return_dict=True)
        ignored = torch.full((labels.size(0), question_embeds.size(1)), -100, dtype=labels.dtype, device=labels.device)
        return MetaLoRAOutput(out.logits, torch.cat([ignored, labels], dim=1), prefix.memory, prefix.layer_memory, prefix.reconstruction, prefix.context_target, prefix.context_mask)

    def forward(self, context_embeds, context_mask, question_embeds, question_mask, answer_embeds, answer_mask, labels, qa_context_indices=None):
        prefix = self.encode_context_prefix(context_embeds, context_mask)
        indices = qa_context_indices if qa_context_indices is not None else torch.arange(question_embeds.size(0), device=question_embeds.device)
        return self.forward_qa_with_prefix(prefix, indices, question_embeds, question_mask, answer_embeds, answer_mask, labels)

    @torch.no_grad()
    def generate_answer_with_prefix(self, prefix, qa_context_indices, question_ids, question_mask, tokenizer, max_new_tokens=128):
        """Generate from a shared context prefix with independent QA caches."""
        embedding = self.qwen.get_input_embeddings()
        rows = []
        for row in range(question_ids.size(0)):
            index = qa_context_indices[row:row + 1].to(question_ids.device, dtype=torch.long)
            cache = self._select_cache(prefix.memory_cache, index, self.qwen.config)
            valid = int(question_mask[row].sum())
            row_ids = question_ids[row][question_mask[row].bool()]
            question = embedding(row_ids[None]).to(dtype=self.dtype)
            qmask = question_mask[row][question_mask[row].bool()][None].bool()
            empty = qmask.new_zeros(1, 0)
            mask = build_continuation_mask(qmask, empty, prefix.memory_length, question.dtype)
            start = prefix.context_length + prefix.memory_length
            positions = torch.arange(start, start + valid, device=question.device).unsqueeze(0)
            output = self.qwen(inputs_embeds=question, attention_mask=mask, position_ids=positions, past_key_values=cache, use_cache=True, return_dict=True)
            cache = output.past_key_values
            generated = []
            next_id = output.logits[:, -1].argmax(dim=-1)
            for step in range(max_new_tokens):
                generated.append(next_id)
                if int(next_id.item()) == int(tokenizer.eos_token_id) or step + 1 == max_new_tokens:
                    break
                token = embedding(next_id[:, None]).to(dtype=self.dtype)
                total_cache_length = prefix.memory_length + valid + step
                attention_mask = torch.ones((1, total_cache_length + 1), dtype=torch.bool, device=token.device)
                position_ids = torch.tensor([[start + valid + step]], device=token.device)
                output = self.qwen(inputs_embeds=token, attention_mask=attention_mask, position_ids=position_ids, past_key_values=cache, use_cache=True, return_dict=True)
                cache = output.past_key_values
                next_id = output.logits[:, -1].argmax(dim=-1)
            rows.append(torch.cat(generated))
        width = max(row.numel() for row in rows)
        result = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long, device=question_ids.device)
        for index, row in enumerate(rows):
            result[index, :row.numel()] = row
        return result

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
    )
    for name,p in model.named_parameters():
        p.requires_grad=("lora_A" in name or "lora_B" in name or name == "memory_tokens" or name.startswith("decoders."))
    return tok, model


QwenMemoryModel = MetaLoRA
ModelOutput = MetaLoRAOutput

__all__ = [
    "MetaLoRA", "MetaLoRAOutput", "QwenMemoryModel", "ModelOutput",
    "StaticLoRALinear", "MemoryDecoder", "ContextPrefix",
    "build_block_causal_mask", "build_continuation_mask",
    "load_model",
]
