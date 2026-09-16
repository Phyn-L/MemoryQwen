"""Skipping the vocabulary head in encode_context_prefix: identical outputs, less memory.

Checks that ``self.qwen(...)`` and ``self._transformer_body(...)`` produce bit-identical
hidden states and KV cache for the prefix, then measures the forward+backward peak.

Run: PYTHONPATH=. python .tmp_analysis/verify_lmhead_skip.py [L]
"""
from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, "/data/lz/MemoryQwen")
from src.model import build_block_causal_mask, load_model  # noqa: E402
from utils.config import TrainConfig  # noqa: E402

L = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
cfg = TrainConfig.from_file("configs/qwen-1.7b/train.yaml")
cfg.validate()
device = torch.device("cuda:0")

tokenizer, model = load_model(cfg)
model.to(device).eval()
body = model._transformer_body
print("resolved body:", type(body).__name__, "| has lm_head:", hasattr(body, "lm_head"), flush=True)
assert not hasattr(body, "lm_head"), "body still exposes an LM head"
assert body is model.qwen.model or "Model" in type(body).__name__

emb = model.qwen.get_input_embeddings()
ids = torch.randint(0, 1000, (1, L), device=device)
with torch.no_grad():
    context = emb(ids)
memory = model.memory_tokens.to(dtype=model.dtype).unsqueeze(0).expand(1, -1, -1)
sequence = torch.cat([context, memory], dim=1)
mask = build_block_causal_mask(torch.ones(1, L, dtype=torch.bool, device=device),
                              memory.size(1), torch.ones(1, 0, dtype=torch.bool, device=device),
                              torch.ones(1, 0, dtype=torch.bool, device=device), sequence.dtype)


def run(tag, module):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    out = module(inputs_embeds=sequence, attention_mask=mask, output_hidden_states=True,
                 use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    fwd = time.time() - t0
    fwd_peak = torch.cuda.max_memory_allocated() / 2 ** 30
    # keep everything alive while measuring, matching a training step's lifetime
    loss = out.hidden_states[-1][:, L:].float().pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    states = tuple(s.detach().clone() for s in out.hidden_states)
    cache = [(layer.keys.detach().clone(), layer.values.detach().clone())
             for layer in out.past_key_values.layers]
    print("  %-22s forward=%5.2fs  peak_forward=%6.2f GiB  peak_fwd+bwd=%6.2f GiB  logits=%s"
          % (tag, fwd, fwd_peak, peak, hasattr(out, "logits")), flush=True)
    del out, loss
    torch.cuda.empty_cache()
    return states, cache


ref_states, ref_cache = run("self.qwen (with head)", model.qwen)
new_states, new_cache = run("_transformer_body", body)

same_states = all(torch.equal(a, b) for a, b in zip(ref_states, new_states)) and len(ref_states) == len(new_states)
same_cache = all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(ref_cache, new_cache))
print("hidden_states identical:", same_states, "| past_key_values identical:", same_cache)
assert same_states and same_cache
print("PASS")
