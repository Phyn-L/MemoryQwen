"""Isolate the LM-head cost: identical loss on the last hidden state, with and without it."""
import sys, time, torch
from transformers import AutoModelForCausalLM
MODEL="/data/lz/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
which, L, B = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
dev=torch.device("cuda:0")
model=AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, local_files_only=True).to(dev).train()
for p in model.parameters(): p.requires_grad_(False)
emb=model.get_input_embeddings()
x=emb(torch.randint(0,1000,(B,L),device=dev)).detach().requires_grad_(True)
module = model if which=="causal_lm" else model.model
mask=torch.ones(B,L,dtype=torch.long,device=dev)
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0=time.time()
out=module(inputs_embeds=x, attention_mask=mask, output_hidden_states=True, use_cache=False, return_dict=True)
out.hidden_states[-1].float().pow(2).mean().backward()
torch.cuda.synchronize()
print("%-12s L=%d B=%d peak=%6.2f GiB %5.2fs"%(which,L,B,torch.cuda.max_memory_allocated()/2**30,time.time()-t0), flush=True)
