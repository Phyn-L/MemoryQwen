import sys, time, torch
sys.path.insert(0,"/data/lz/MemoryQwen"); sys.path.insert(0,"/data/lz/MemoryQwen/.tmp_analysis")
from src.model import build_block_causal_mask, build_continuation_mask
from bench_eng import block_mask_fast, continuation_mask_fast
dev=torch.device("cuda:0")
print("B=2 M=8 QL=64 AL=32")
for L in (128,256,512,1024,2048):
    c=torch.ones(2,L,dtype=torch.bool,device=dev); q=torch.ones(2,64,dtype=torch.bool,device=dev); a=torch.ones(2,32,dtype=torch.bool,device=dev)
    row=[]
    for name,fn in (("old",lambda: build_block_causal_mask(c,8,q,a,torch.bfloat16)),("new",lambda: block_mask_fast(c,8,q,a,torch.bfloat16))):
        fn(); torch.cuda.synchronize(); t0=time.time()
        n = max(1, min(20, int(2.0/max(1e-6,0))))
        for _ in range(5): fn()
        torch.cuda.synchronize(); row.append("%s=%7.2fms"%(name,(time.time()-t0)/5*1000))
    print("  L=%5d  %s"%(L," ".join(row)), flush=True)
print("continuation mask B=8 M=8 QL=64 AL=32")
q=torch.ones(8,64,dtype=torch.bool,device=dev); a=torch.ones(8,32,dtype=torch.bool,device=dev)
for name,fn in (("old",lambda: build_continuation_mask(q,a,8,torch.bfloat16)),("new",lambda: continuation_mask_fast(q,a,8,torch.bfloat16))):
    fn(); torch.cuda.synchronize(); t0=time.time()
    for _ in range(20): fn()
    torch.cuda.synchronize(); print("  %s=%7.2fms"%(name,(time.time()-t0)/20*1000), flush=True)
