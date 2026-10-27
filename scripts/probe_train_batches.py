"""Dump raw records and collated shapes for a deterministic training-step window."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from src.data.pipeline import make_collate, make_context_dataset
from src.data.sortish_sampler import SortishSampler
from utils.config import TrainConfig


def _jsonable_record(record):
    return {
        "id": record.context_id,
        "dataset": record.dataset,
        "context": record.context,
        "qa_pairs": [
            {
                "id": turn.context_id,
                "question": turn.question,
                "answer": turn.answer,
                "messages": [
                    {"role": item.role, "content": item.content}
                    for item in turn.messages
                ],
            }
            for turn in record.turns()
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ablation/h200_train_baseline.yaml")
    parser.add_argument("--machine", default="h200")
    parser.add_argument("--start-step", type=int, default=36500)
    parser.add_argument("--end-step", type=int, default=36714)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--output", default="outputs/oom_probe_36500_36714.jsonl")
    args = parser.parse_args()
    if args.start_step < 0 or args.end_step < args.start_step:
        parser.error("invalid step range")
    if args.world_size <= 0:
        parser.error("world-size must be positive")

    cfg = TrainConfig.from_file(args.config, machine=args.machine)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name_or_path, local_files_only=True
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    dataset = make_context_dataset(
        cfg, "train", tokenizer, limit=None, num_workers=0
    )
    sampler = SortishSampler(
        dataset,
        tokenizer,
        cfg.training.batch_size,
        cfg.data.sortish_bucket_multiplier,
        cfg.training.seed,
        cfg.data.chat_template_enable_thinking,
        None,
    )
    # Reproduce the DataLoader's post-sampler batch order, then seek directly to the
    # requested global optimizer-step window. No Dataset rows are collated before start_step.
    batches = list(sampler.buckets)
    random.Random(cfg.training.seed + sampler.epoch).shuffle(batches)
    start_batch = args.start_step * args.world_size
    end_batch = (args.end_step + 1) * args.world_size
    if end_batch > len(batches):
        parser.error(
            f"step window ends at batch {end_batch}, but sampler has only {len(batches)} batches"
        )
    batches = batches[start_batch:end_batch]
    print(
        f"seeking directly to optimizer step {args.start_step}; "
        f"processing {len(batches)} rank batches (no earlier DataLoader batches)",
        flush=True,
    )
    collate = make_collate(cfg, tokenizer, sample_qa=True, training=True)
    collate.set_epoch(0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for step in range(args.start_step, args.end_step + 1):
            for rank in range(args.world_size):
                batch_number = (step - args.start_step) * args.world_size + rank
                indices = batches[batch_number]
                raw_rows = [dataset[index] for index in indices]
                batch = collate(raw_rows)
                tensor_names = (
                    "context_ids", "context_mask", "qa_input_ids", "qa_input_mask",
                    "qa_prefix_ids", "qa_prefix_mask", "qa_answer_prefix_ids",
                    "qa_answer_prefix_mask", "qa_target_ids", "qa_target_mask",
                    "recon_target_ids", "recon_target_mask",
                )
                payload = {
                    "step": step,
                    "rank": rank,
                    "batch_indices": indices,
                    "context_records": [_jsonable_record(row) for row in raw_rows],
                    "shapes": {name: list(batch[name].shape) for name in tensor_names},
                    "valid_lengths": {
                        name: batch[name].sum(dim=1).tolist()
                        for name in (
                            "context_mask", "qa_input_mask", "qa_prefix_mask",
                            "qa_answer_prefix_mask", "qa_target_mask",
                            "recon_target_mask",
                        )
                    },
                    "qa_context_indices": batch["qa_context_indices"].tolist(),
                    "recon_context_indices": batch["recon_context_indices"].tolist(),
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
