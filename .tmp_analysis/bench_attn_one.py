"""One variant per process (clean allocator) so the memory numbers are comparable.

Usage: python bench_attn_one.py <variant> [L] [B]
Variants: causal_lm_2d_hs causal_lm_none_hs causal_lm_4d_hs causal_lm_2d_nohs
          base_2d_hs base_2d_nohs base_none_hs
"""
from __future__ import annotations

import sys
import time

import torch
from transformers import AutoModelForCausalLM

MODEL = "/data/lz/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
variant = sys.argv[1]
L = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
B = int(sys.argv[3]) if len(sys.argv) > 3 else 1
dev = torch.device("cuda:0")

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, local_files_only=True)
model.to(dev).train()
for p in model.parameters():
    p.requires_grad_(False)          # freeze: we only measure activation memory
emb = model.get_input_embeddings()
x = emb(torch.randint(0, 1000, (B, L), device=dev)).detach().requires_grad_(True)

module = model.model if variant.startswith("base") else model
hs = variant.endswith("_hs")
if "none" in variant:
    mask = None
elif "4d" in variant:
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
    mask = torch.zeros(B, 1, L, L, dtype=torch.bfloat16, device=dev).masked_fill(
        ~causal[None, None], torch.finfo(torch.bfloat16).min)
else:
    mask = torch.ones(B, L, dtype=torch.long, device=dev)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
t0 = time.time()
out = module(inputs_embeds=x, attention_mask=mask, output_hidden_states=hs,
             use_cache=False, return_dict=True)
y = out.logits if hasattr(out, "logits") else out.last_hidden_state
y.float().pow(2).mean().backward()
torch.cuda.synchronize()
print("%-22s L=%d B=%d  peak=%6.2f GiB  %5.2f s  hidden_states_kept=%s"
      % (variant, L, B, torch.cuda.max_memory_allocated() / 2 ** 30, time.time() - t0,
         len(out.hidden_states) if out.hidden_states is not None else 0), flush=True)
