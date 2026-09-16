"""Exercise the checkpoint validation: happy path, resized model, missing head, stray key."""
import sys, copy, torch
sys.path.insert(0, "/data/lz/MemoryQwen")
from transformers import Qwen3Config, Qwen3ForCausalLM
from src.model import MetaLoRA
from utils.checkpoint import CheckpointManager

COMMON = dict(rank=4, alpha=8.0, memory_length=8, decoder_hidden_size=32, decoder_heads=4,
              decoder_ffn_ratio=2, max_context_tokens=128,
              target_modules=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"))

def tiny(**over):
    cfg = Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=128)
    kw = dict(COMMON); kw.update(over)
    m = MetaLoRA(Qwen3ForCausalLM(cfg), context_lm=True, **kw)
    for n, p in m.named_parameters():
        p.requires_grad = ("lora_" in n or n == "memory_tokens" or n.startswith(("decoders.","context_lm_head.")))
    return m

import tempfile, pathlib
d = pathlib.Path(tempfile.mkdtemp())
mgr = CheckpointManager(d)
src = tiny(); opt = torch.optim.AdamW([p for p in src.parameters() if p.requires_grad], lr=1e-4)
path = mgr.save(src, opt, opt, 7, {})

def case(name, model, **kw):
    try:
        step, _ = mgr.load(path, model, **kw)
        print(f"  {name}: loaded step={step}")
    except RuntimeError as exc:
        print(f"  {name}: RuntimeError -> {str(exc)[:110]}...")

print("1) identical model")
case("ok", tiny())

print("2) memory_length 8 -> 16 (shape mismatch)")
case("mismatch", tiny(memory_length=16))

print("3) checkpoint stripped of a trainable tensor")
state = torch.load(path, map_location="cpu", weights_only=False)
stripped = copy.deepcopy(state); stripped["model"].pop("context_lm_head.weight")
p2 = d / "stripped.pt"; torch.save(stripped, p2)
for flag in (False, True):
    try:
        step, _ = mgr.load(p2, tiny(), allow_missing_trainable=flag)
        print(f"  allow_missing_trainable={flag}: loaded step={step}")
    except RuntimeError as exc:
        print(f"  allow_missing_trainable={flag}: RuntimeError -> {str(exc)[:90]}...")

print("4) stray tensor in the file")
stray = copy.deepcopy(state); stray["model"]["bogus.weight"] = torch.zeros(2, 2)
p3 = d / "stray.pt"; torch.save(stray, p3)
mgr2 = CheckpointManager(d)
try:
    mgr2.load(p3, tiny())
    print("  unexpected: loaded (BAD)")
except RuntimeError as exc:
    print(f"  unexpected: RuntimeError -> {str(exc)[:110]}...")
