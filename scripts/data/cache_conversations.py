#!/usr/bin/env python3
"""Build the conversation and sortish caches before a training run.

This command deliberately does not load the Qwen weights.  It only loads the local
tokenizer and runs the same filtering/indexing path as ``scripts/train.py``.  The
result is reusable by later runs with the same source file, tokenizer and data limits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from src.data import SortishSampler
from src.pipeline import data_cache_dir, make_context_dataset, resolve_split_limit
from utils.config import TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_baseline.yaml")
    parser.add_argument("--machine", default=None)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "both"),
        default="both",
        help="Which split to preprocess; train also builds sortish lengths.",
    )
    parser.add_argument("--max-context-tokens", type=int)
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--max-question-tokens", type=int)
    parser.add_argument("--max-answer-tokens", type=int)
    parser.add_argument("--tokenizer-filter-batch-size", type=int)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes for conversation filtering; defaults to data.num_workers.",
    )
    parser.add_argument("--train-max-samples", type=int)
    parser.add_argument("--validation-max-samples", type=int)
    args = parser.parse_args()

    cfg = TrainConfig.from_file(args.config, machine=args.machine)
    if args.max_context_tokens is not None:
        cfg.data.max_context_tokens = args.max_context_tokens
    if args.max_input_tokens is not None:
        cfg.data.max_input_tokens = args.max_input_tokens
    if args.max_question_tokens is not None:
        cfg.data.max_question_tokens = args.max_question_tokens
    if args.max_answer_tokens is not None:
        cfg.data.max_answer_tokens = args.max_answer_tokens
    if args.tokenizer_filter_batch_size is not None:
        cfg.data.tokenizer_filter_batch_size = args.tokenizer_filter_batch_size
    if args.train_max_samples is not None:
        cfg.data.train_max_samples = args.train_max_samples
    if args.validation_max_samples is not None:
        cfg.data.validation_max_samples = args.validation_max_samples
    num_workers = cfg.data.train_num_workers if args.num_workers is None else args.num_workers
    if num_workers < 0:
        raise SystemExit("--num-workers must be non-negative")
    cfg.validate()
    if not cfg.data.cache_filtered_dataset and not cfg.data.cache_sortish_lengths:
        raise SystemExit(
            "Both data.cache_filtered_dataset and data.cache_sortish_lengths are false; "
            "there is nothing to cache."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name_or_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    cache_dir = data_cache_dir(cfg)
    splits = ("train", "validation") if args.split == "both" else (args.split,)

    for split in splits:
        print(
            f"[cache] preparing {split} split from {cfg.data.split_file(split)}",
            flush=True,
        )
        dataset = make_context_dataset(
            cfg,
            split,
            tokenizer,
            limit=resolve_split_limit(cfg, split),
            allow_empty=(split == "validation"),
            num_workers=num_workers,
        )
        print(f"[cache] {split}: {len(dataset)} usable contexts", flush=True)
        if split == "train" and cfg.data.cache_sortish_lengths:
            SortishSampler(
                dataset,
                tokenizer,
                cfg.training.batch_size,
                cfg.data.sortish_bucket_multiplier,
                cfg.training.seed,
                cfg.data.chat_template_enable_thinking,
                cache_dir,
                num_workers,
            )
            print(f"[cache] train sortish lengths ready under {cache_dir}", flush=True)

    print(
        f"[cache] complete; reusable cache root: {Path(cache_dir).resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
