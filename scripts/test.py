from __future__ import annotations

import argparse
import torch
from torch.utils.data import DataLoader

from src.evaluator import Evaluator
from src.data import AggregatedQADataset, collate_fn
from src.model import load_model
from utils.config import TrainConfig
from utils.checkpoint import CheckpointManager


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Qwen memory checkpoint")
    parser.add_argument("--config", default="configs/qwen-1.7b/train.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    cfg = TrainConfig.from_file(args.config); cfg.validate()
    tokenizer, model = load_model(cfg)
    CheckpointManager(cfg.checkpoint.output_dir).load(args.checkpoint, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    names = getattr(cfg.data, f"{args.split}_datasets") or cfg.data.dataset
    split_name = getattr(cfg.data, f"{args.split}_split")
    ds = AggregatedQADataset(cfg.data.root, names, split_name, tokenizer, cfg.data.max_context_tokens, args.max_samples, cfg.data.filter_long_context, cfg.data.filter_no_qa, allow_empty=True, cache_dir=cfg.checkpoint.output_dir if cfg.data.cache_dataset else None)
    collate = lambda rows: collate_fn(rows, tokenizer, cfg.data.max_context_tokens, cfg.data.max_question_tokens, cfg.data.max_answer_tokens, cfg.data.append_eos, cfg.data.use_chat_template, cfg.data.chat_template_enable_thinking)
    loader = DataLoader(
        ds, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate,
    )
    if not len(loader):
        print(f"No usable records in {args.split}; skipping evaluation.")
        return
    evaluator = Evaluator(tokenizer, cfg)
    print(evaluator.teacher_forced(model, loader, device))
    print(evaluator.autoregressive(model, loader, device))


if __name__ == "__main__":
    main()
