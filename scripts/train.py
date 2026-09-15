from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
import random

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.config import TrainConfig, dtype_from_name
from src.data import AggregatedQADataset, SortishSampler, collate_fn
from src.evaluator import Evaluator
from src.losses import (
    combine_losses,
    memory_contrastive_loss,
    qa_loss,
    reconstruction_loss,
)
from src.model import load_model
from utils import CheckpointManager, build_optimizer, build_scheduler


def _model_run_label(model_path: str) -> str:
    match = re.search(r"Qwen(?:3)?[-_]?([0-9]+(?:\.[0-9]+)?)[Bb]", model_path, re.IGNORECASE)
    if match:
        return f"Qwen{match.group(1)}B"
    return "Qwen"


def _new_run_name(model_path: str) -> str:
    return f"{_model_run_label(model_path)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _model_cache_dir(model_path: str) -> Path:
    return Path("outputs") / _model_run_label(model_path)


def _configure_run_paths(cfg: TrainConfig, accelerator, resume: str | None) -> None:
    if resume:
        run_dir = Path(resume).resolve().parent
        cfg.checkpoint.output_dir = str(run_dir)
        cfg.logging.wandb_run_name = cfg.logging.wandb_run_name or run_dir.name
        return
    is_main = accelerator is None or accelerator.is_main_process
    run_name = _new_run_name(cfg.model.name_or_path) if is_main else None
    num_processes = accelerator.num_processes if accelerator is not None else 1
    if num_processes > 1:
        names = [run_name]
        torch.distributed.broadcast_object_list(names, src=0)
        run_name = names[0]
    cfg.logging.wandb_run_name = run_name
    cfg.checkpoint.output_dir = str(Path("outputs") / run_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen-1.7b/train.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--wandb-run-id")
    args = parser.parse_args()

    cfg = TrainConfig.from_file(args.config)
    cfg.validate()
    random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    tokenizer, model = load_model(cfg)

    accelerator = None
    try:
        from accelerate import Accelerator
        configured_dtype = dtype_from_name(cfg.model.torch_dtype)
        mixed_precision = {
            torch.float32: "no",
            torch.float16: "fp16",
            torch.bfloat16: "bf16",
        }[configured_dtype]
        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision=mixed_precision if torch.cuda.is_available() else "no",
        )
        device = accelerator.device
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    _configure_run_paths(cfg, accelerator, args.resume)
    model_cache_dir = _model_cache_dir(cfg.model.name_or_path)

    def make_dataset(split: str):
        names = getattr(cfg.data, f"{split}_datasets") or cfg.data.dataset
        split_name = getattr(cfg.data, f"{split}_split")
        limit = getattr(cfg.data, f"{split}_max_samples")
        if limit is None and split == "train":
            limit = cfg.data.max_samples
        return AggregatedQADataset(
            cfg.data.root, names, split_name, tokenizer,
            cfg.data.max_context_tokens, limit,
            cfg.data.filter_long_context, cfg.data.filter_no_qa, allow_empty=(split != "train"),
            cache_dir=model_cache_dir if cfg.data.cache_dataset else None,
        )

    collate = lambda rows: collate_fn(
        rows, tokenizer, cfg.data.max_context_tokens,
        cfg.data.max_question_tokens, cfg.data.max_answer_tokens, cfg.data.append_eos,
        cfg.data.use_chat_template, cfg.data.chat_template_enable_thinking,
    )
    train_dataset = make_dataset("train")
    sampler = SortishSampler(
        train_dataset, tokenizer, cfg.training.batch_size,
        cfg.data.sortish_bucket_multiplier, cfg.training.seed, cfg.data.use_chat_template,
        cfg.data.chat_template_enable_thinking,
        model_cache_dir if cfg.data.cache_sortish_lengths else None,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, sampler=sampler, collate_fn=collate,
    )
    validation_dataset = make_dataset("validation")
    validation_loader = DataLoader(
        validation_dataset, batch_size=cfg.training.batch_size,
        shuffle=False, collate_fn=collate,
    )
    evaluator = Evaluator(tokenizer, cfg)
    total_steps = max(
        1,
        (len(train_loader) * cfg.training.epochs + cfg.training.grad_accumulation - 1)
        // cfg.training.grad_accumulation,
    )
    optimizer = build_optimizer(model, cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg.scheduler, total_steps)
    manager = CheckpointManager(
        cfg.checkpoint.output_dir, cfg.checkpoint.save_every_steps,
    )

    step = 0
    if args.resume:
        step, saved_run_id = manager.load(args.resume, model, optimizer, scheduler)
        if not cfg.logging.wandb_run_id:
            cfg.logging.wandb_run_id = saved_run_id
    if args.wandb_run_id:
        cfg.logging.wandb_run_id = args.wandb_run_id

    if accelerator is not None:
        model, optimizer, scheduler, train_loader, validation_loader = accelerator.prepare(
            model, optimizer, scheduler, train_loader, validation_loader,
        )

    run = None
    if accelerator is None or accelerator.is_main_process:
        try:
            import wandb
            run = wandb.init(
                project=cfg.logging.wandb_project, name=cfg.logging.wandb_run_name,
                id=cfg.logging.wandb_run_id,
                resume="allow" if cfg.logging.wandb_run_id else None,
                mode=cfg.logging.wandb_mode, config=cfg.to_dict(),
            )
        except ImportError:
            pass

    model.train()
    progress = tqdm(
        total=total_steps, desc="Training", unit="step",
        disable=not (accelerator is None or accelerator.is_main_process),
        initial=step,
    )
    for epoch in range(cfg.training.epochs):
        sampler.set_epoch(epoch)
        for batch in train_loader:
            ids = {k: v.to(device) for k, v in batch.items() if k != "records"}
            base_model = accelerator.unwrap_model(model) if accelerator is not None else model
            embedding = base_model.qwen.get_input_embeddings()
            output = model(
                embedding(ids["context_ids"]), ids["context_mask"],
                embedding(ids["question_ids"]), ids["question_mask"],
                embedding(ids["answer_ids"]), ids["answer_mask"], ids["labels"],
            )
            qa = qa_loss(output.logits, output.labels)
            reconstruction = reconstruction_loss(
                output.reconstruction, output.context_target, output.context_mask,
                cfg.memory.reconstruction_cosine_weight, cfg.memory.reconstruction_loss,
            )
            contrastive = (
                memory_contrastive_loss(output.memory, torch.roll(output.memory, shifts=1, dims=0), cfg.memory.contrastive_temperature, cfg.memory.contrastive_margin)
                if cfg.memory.contrastive_weight else None
            )
            total, terms = combine_losses(
                qa, reconstruction, cfg.memory.qa_weight,
                cfg.memory.reconstruction_weight, contrastive,
                cfg.memory.contrastive_weight,
            )
            if accelerator is not None:
                accelerator.backward(total / cfg.training.grad_accumulation)
            else:
                (total / cfg.training.grad_accumulation).backward()
            if (step + 1) % cfg.training.grad_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.training.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            progress.update(1)
            progress.set_postfix(
                loss=f"{float(total.detach()):.4f}",
                qa=f"{float(qa.detach()):.4f}",
                recon=f"{float(reconstruction.detach()):.4f}",
            )

            if run and step % 10 == 0:
                run.log({f"train/{k}": float(v.detach()) for k, v in terms.items()}, step=step)
            is_main = accelerator is None or accelerator.is_main_process
            if is_main and len(validation_loader) and step % cfg.evaluation.teacher_forced_every == 0:
                metrics = evaluator.teacher_forced(base_model, validation_loader, device)
                if run:
                    run.log({f"val/teacher_forced/{k}": v for k, v in metrics.items()}, step=step)
            if is_main and len(validation_loader) and step % cfg.evaluation.autoregressive_every == 0:
                metrics = evaluator.autoregressive(base_model, validation_loader, device)
                if run:
                    run.log({f"val/autoregressive/{k}": v for k, v in metrics.items()}, step=step)
            if step % cfg.checkpoint.save_every_steps == 0 and is_main:
                checkpoint_model = accelerator.unwrap_model(model) if accelerator is not None else model
                manager.save(checkpoint_model, optimizer, scheduler, step, cfg.to_dict(), wandb_run_id=getattr(run, "id", None))

    if accelerator is None or accelerator.is_main_process:
        checkpoint_model = accelerator.unwrap_model(model) if accelerator is not None else model
        manager.save(checkpoint_model, optimizer, scheduler, step, cfg.to_dict(), final=True, wandb_run_id=getattr(run, "id", None))
    progress.close()
    if run:
        run.finish()


if __name__ == "__main__":
    main()
