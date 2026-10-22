"""Synthetic writer/QA smoke test, optionally under torchrun."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import trange
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses import objective_total, supervision_objectives
from src.model import ContextMemoryModel, load_model
from utils.config import TrainConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_baseline.yaml")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.device == "cuda":
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" else "cpu")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    torch.manual_seed(42)
    cfg = TrainConfig.from_file(args.config)
    if args.tiny:
        qcfg = Qwen3Config(
            vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=40960, tie_word_embeddings=True,
        )
        qcfg._attn_implementation = "sdpa"
        model = ContextMemoryModel(
            Qwen3ForCausalLM(qcfg), rank=2, memory_length=8,
            context_layer_indices=(-1, -2), memory_resampler_hidden_size=32,
            readout_length=2, question_resampler_hidden_size=32,
            context_gradient_checkpointing=True,
            recon_user_token_ids=(2,), recon_answer_prefix_token_ids=(3, 4),
        )
    else:
        _, model = load_model(cfg)
    model.to(device).train(args.backward)
    length = args.length or model.max_context_tokens
    ids = torch.randint(2, model.qwen.config.vocab_size, (1, length), device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    embedding = model.qwen.get_input_embeddings()
    context = embedding(ids)
    question_ids = torch.randint(2, model.qwen.config.vocab_size, (1, 8), device=device)
    answer_ids = torch.randint(2, model.qwen.config.vocab_size, (1, 4), device=device)
    question, answer = embedding(question_ids), embedding(answer_ids)
    wrapped = DistributedDataParallel(
        model, device_ids=[local_rank] if device.type == "cuda" else None,
    ) if world > 1 and args.backward else model
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    for step in trange(args.steps, desc="context memory", disable=rank != 0):
        if args.backward:
            optimizer.zero_grad(set_to_none=True)
            # Both supervision branches the trainer runs: the QA row and the Recon row that
            # reconstructs the same context. Without the Recon row ``recon_task_embedding``
            # would never receive a gradient, which is what this smoke test asserts.
            batch_contexts = context.size(0)
            recon_batch = {
                "context_indices": torch.arange(batch_contexts, device=device),
                "context_embeds": context,
                "context_mask": mask,
                "labels": ids,
            }
            generator = torch.Generator(device=device.type).manual_seed(
                cfg.training.seed + step
            )
            output = wrapped(
                context, mask, question, torch.ones_like(question_ids, dtype=torch.bool),
                answer, torch.ones_like(answer_ids, dtype=torch.bool), answer_ids,
                context_ids=ids, loss_config=cfg.loss, loss_only=True,
                recon_batch=recon_batch, generator=generator,
            )
            # Same objective the trainer builds: per-row means, both tasks, distill on top.
            task_ids = torch.cat([
                torch.ones(batch_contexts, dtype=torch.long, device=device),
                torch.zeros(batch_contexts, dtype=torch.long, device=device),
            ])
            supervised, terms = supervision_objectives(
                output.supervision_losses,
                task_ids,
                cfg.loss,
                row_mask=output.supervision_mask,
            )
            loss = supervised
            if output.objective_terms:
                loss = loss + objective_total(supervised * 0, output.objective_terms, cfg.loss)
            assert torch.isfinite(loss), loss
            assert torch.isfinite(terms["qa_loss"]) and torch.isfinite(terms["recon_loss"]), terms
            loss.backward()
            missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
            assert not missing, missing
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            assert model.context_resampler.latents.grad.abs().sum() > 0
            optimizer.step()
        else:
            with torch.no_grad():
                prefix = model.encode_context_prefix(context, mask)
                memory_length = model.context_resampler.num_memory
                assert prefix.memory.shape == (1, memory_length, model.qwen_hidden_size)
                assert torch.isfinite(prefix.memory).all()
                assert prefix.memory.shape[1] == memory_length
    if device.type == "cuda":
        torch.cuda.synchronize()
    report = {
        "device": str(device), "world_size": world, "tiny": args.tiny,
        "context_length": length, "memory_length": model.context_resampler.num_memory,
        "backward": args.backward, "steps": args.steps,
        "seconds": time.perf_counter() - started,
        "torch": torch.__version__,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None,
    }
    if rank == 0:
        print(json.dumps(report, indent=2))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
