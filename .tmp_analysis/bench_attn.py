"""Why does a 2048-token step OOM on a 24 GiB card?

For one Qwen3-1.7B forward+backward at L=2048, B=1, compares:
  which module is called      -> CausalLM (computes 151936-way logits) vs base model
  which attention mask        -> 2-D padding / None / 4-D additive
  output_hidden_states        -> True / False

Also times the repo's mask builders against vectorised replacements for several L.

Run: PYTHONPATH=. python .tmp_analysis/bench_attn.py [L] [B]
"""
from __future__ import annotations

import sys
import time

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, "/data/lz/MemoryQwen")
from src.model import build_block_causal_mask, build_continuation_mask  # noqa: E402
sys.path.insert(0, "/data/lz/MemoryQwen/.tmp_analysis")
from bench_eng import block_mask_fast, continuation_mask_fast  # noqa: E402

MODEL = "/data/lz/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
L = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
B = int(sys.argv[2]) if len(sys.argv) > 2 else 1
dev = torch.device("cuda:0")

torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, local_files_only=True)
model.to(dev).train()
print("attn_implementation =", model.config._attn_implementation, "| L =", L, "| B =", B, flush=True)
emb = model.get_input_embeddings()
ids = torch.randint(0, 1000, (B, L), device=dev)
x = emb(ids)
causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
two_d = torch.ones(B, L, dtype=torch.long, device=dev)
four_d = torch.zeros(B, 1, L, L, dtype=torch.bfloat16, device=dev).masked_fill(
    ~causal[None, None], torch.finfo(torch.bfloat16).min)


def run(tag, module, mask, hidden_states):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    torch.cuda.synchronize()
    t0 = time.time()
    try:
        out = module(inputs_embeds=x, attention_mask=mask, output_hidden_states=hidden_states,
                     use_cache=False, return_dict=True)
        loss = (out.logits if hasattr(out, "logits") else out.last_hidden_state).float().pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        print("  %-46s peak=%6.2f GiB  %5.2f s" % (tag, (torch.cuda.max_memory_allocated() - base) / 2 ** 30, time.time() - t0), flush=True)
        del out, loss
    except torch.cuda.OutOfMemoryError:
        torch.cuda.synchronize()
        print("  %-46s OOM" % tag, flush=True)
    finally:
        torch.cuda.empty_cache()


print("--- forward + backward", flush=True)
run("CausalLM  2-D mask   hs=True", model, two_d, True)
run("CausalLM  mask=None  hs=True", model, None, True)
run("CausalLM  4-D add.   hs=True", model, four_d, True)
run("CausalLM  2-D mask   hs=False", model, two_d, False)
run("base      2-D mask   hs=True", model.model, two_d, True)
run("base      2-D mask   hs=False", model.model, two_d, False)
run("base      mask=None  hs=False", model.model, None, False)

# ---------------------------------------------------------------- mask construction vs L
print("--- mask build time vs context length (B=2, M=8, QL=64, AL=32)", flush=True)
for length in (128, 256, 512, 1024, 2048):
    c = torch.ones(2, length, dtype=torch.bool, device=dev)
    q = torch.ones(2, 64, dtype=torch.bool, device=dev)
    a = torch.ones(2, 32, dtype=torch.bool, device=dev)
    for name, fn in (("old", lambda: build_block_causal_mask(c, 8, q, a, torch.bfloat16)),
                     ("new", lambda: block_mask_fast(c, 8, q, a, torch.bfloat16))):
        fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        print("  L=%5d %s  %8.2f ms/call" % (length, name, (time.time() - t0) / 5 * 1000), flush=True)
