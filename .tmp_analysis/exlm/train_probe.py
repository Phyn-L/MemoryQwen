"""Short real-model probe of the context_lm objective: cost + learning signal."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ROOT = Path("/data/lz/MemoryQwen")
sys.path.insert(0, str(ROOT))

from src.data import AggregatedContextDataset, SortishSampler, collate_fn
from src.losses import context_lm_loss, qa_loss, reconstruction_loss
from src.model import load_model
from utils.config import TrainConfig

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=str(ROOT / "configs/qwen-1.7b/train.yaml"))
ap.add_argument("--mode", choices=("context_lm", "mse_cosine"), required=True)
ap.add_argument("--steps", type=int, default=120)
ap.add_argument("--gpu", default="0")
ap.add_argument("--context-lm-positions", type=int, default=256)
args = ap.parse_args()

cfg = TrainConfig.from_file(args.config)
cfg.memory.reconstruction_loss = args.mode
cfg.memory.context_lm_positions = args.context_lm_positions
cfg.validate()
device = torch.device("cuda:0")
tokenizer, model = load_model(cfg)
model.to(device).train()
embed = model.qwen.get_input_embeddings()

train_ds = AggregatedContextDataset(cfg.data.root, ["squad"], "train", tokenizer,
                                    cfg.data.max_context_tokens, 256,
                                    cfg.data.filter_long_context, cfg.data.filter_no_qa,
                                    cache_dir=Path("outputs") / "Qwen1.7B")
sampler = SortishSampler(train_ds, tokenizer, 1, cfg.data.sortish_bucket_multiplier, 42,
                         False, False, Path("outputs") / "Qwen1.7B")
collate = lambda rows: collate_fn(rows, tokenizer, max_context_tokens=cfg.data.max_context_tokens,
                                  max_question_tokens=cfg.data.max_question_tokens,
                                  max_answer_tokens=cfg.data.max_answer_tokens,
                                  qa_per_context=cfg.data.qa_per_context, sample_qa=True,
                                  question_padding_side=cfg.data.question_padding_side,
                                  eos_mode=cfg.data.eos_mode)
loader = DataLoader(train_ds, batch_size=1, sampler=sampler, collate_fn=collate)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
torch.cuda.reset_peak_memory_stats()
print(f"mode={args.mode} trainable={n_params/1e6:.2f}M "
      f"head={'yes' if model.context_lm_head is not None else 'no'}", flush=True)

rows = []
t0 = time.time()
for step, batch in enumerate(loader):
    if step >= args.steps:
        break
    ids = {k: v.to(device) for k, v in batch.items() if k != "records"}
    out = model(embed(ids["context_ids"]), ids["context_mask"],
                embed(ids["question_ids"]), ids["question_mask"],
                embed(ids["answer_ids"]), ids["answer_mask"], ids["labels"],
                ids["qa_context_indices"], context_ids=ids["context_ids"],
                context_lm_positions=cfg.memory.context_lm_positions)
    qa = qa_loss(out.logits, out.labels)
    if args.mode == "context_lm":
        aux = context_lm_loss(out.context_lm_hidden, out.context_lm_labels,
                              model.context_lm_head, out.context_lm_mask)
    else:
        aux = reconstruction_loss(out.reconstruction, out.context_target, out.context_mask,
                                  cfg.memory.reconstruction_cosine_weight, "mse_cosine")
    (qa + aux).backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    if step % 10 == 0 or step == args.steps - 1:
        rows.append((step, float(qa), float(aux)))
        print(f"  step {step:4d} qa={float(qa):7.4f} aux={float(aux):7.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

elapsed = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 2**30
print(f"RESULT mode={args.mode} steps={len(rows)} time={elapsed:.1f}s "
      f"per_step={elapsed/max(1,len(rows)):.3f}s peak_gib={peak:.2f} "
      f"first_aux={rows[0][2]:.4f} last_aux={rows[-1][2]:.4f} "
      f"first_qa={rows[0][1]:.4f} last_qa={rows[-1][1]:.4f}")
