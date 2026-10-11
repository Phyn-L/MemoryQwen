"""Find the largest per-GPU training batch that fits for conversation QA."""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

# ``python scripts/<file>.py`` puts ``scripts/`` (rather than the repository
# root) on ``sys.path``. Make direct execution behave like ``python -m``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model import load_model
from src.pipeline import make_collate, make_context_dataset
from src.data import ContextRecord, ConversationMessage
from utils.config import TrainConfig


DATASETS = {
    "c1qa": ("configs/conversation_c1qa.yaml", "ift_c1qa.jsonl"),
    "pwc": ("configs/conversation_pwc.yaml", "ift_pwc.jsonl"),
}


def configure(config_path: str, source: Path, rows: int) -> TrainConfig:
    cfg = TrainConfig.from_file(config_path)
    for name in ("conversation_file", "train_conversation_file",
                 "validation_conversation_file", "test_conversation_file"):
        setattr(cfg.data, name, str(source))
    cfg.data.max_samples = rows
    cfg.data.train_max_samples = rows
    # Explicit stress-test budget: context / question / answer.
    cfg.data.max_context_tokens = 2048
    cfg.data.max_conversation_tokens = 128
    cfg.data.max_answer_tokens = 64
    cfg.data.train_tasks = "qa"
    cfg.memory.token_recon_weight = 0.0
    cfg.memory.causal_recon_weight = 0.0
    cfg.memory.distill_weight = 0.0
    cfg.memory.qa_weight = 1.0
    cfg.memory.recon_weight = 0.0
    # Read the first rows and let the normal collator truncate them to the
    # configured training limits instead of scanning for short examples.
    cfg.data.filter_long_context = False
    cfg.validate()
    return cfg


def extreme_records(count: int):
    """Build records that saturate the configured context/question/answer limits."""
    context = "context_token " * 4096
    question = "question_token " * 4096
    answer = "answer_token " * 1024
    messages = (
        ConversationMessage("user", question),
        ConversationMessage("assistant", answer),
    )
    return tuple(
        ContextRecord(context, messages, dataset="synthetic_extreme", context_id=str(index))
        for index in range(count)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/data/lz/contexts")
    parser.add_argument("--datasets", default="c1qa,pwc")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument(
        "--sample-mode",
        choices=("extreme", "source"),
        default="extreme",
        help="Use max-length synthetic records (default) or source JSONL records",
    )
    parser.add_argument("--model-config", default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run outside the sandbox on a real 4090")
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    if not batch_sizes or any(value <= 0 for value in batch_sizes):
        raise ValueError("--batch-sizes must contain positive integers")
    names = [value.strip().lower() for value in args.datasets.split(",") if value.strip()]
    device = torch.device("cuda")

    for dataset_name in names:
        config_path, filename = DATASETS[dataset_name]
        if args.model_config:
            config_path = args.model_config
        cfg = configure(config_path, Path(args.data_root) / filename, args.rows)
        tokenizer, model = load_model(cfg)
        model.to(device).train()
        embedding = model.qwen.get_input_embeddings()
        collate = make_collate(cfg, tokenizer, sample_qa=True, training=True)
        if args.sample_mode == "extreme":
            rows = list(extreme_records(args.rows))
        else:
            dataset = make_context_dataset(cfg, "train", tokenizer, limit=args.rows)
            rows = [dataset[index] for index in range(len(dataset))]
        print(f"[{dataset_name}] model={cfg.model.name_or_path}")
        print(f"[{dataset_name}] sample_mode={args.sample_mode} rows={len(rows)}")
        print(f"[{dataset_name}] limits=context={cfg.data.max_context_tokens}, "
              f"question={cfg.data.max_conversation_tokens}, answer={cfg.data.max_answer_tokens}")
        for batch_size in batch_sizes:
            if batch_size > len(rows):
                print(f"[{dataset_name}] bs={batch_size}: skipped (only {len(rows)} rows loaded)")
                continue
            batch = collate(rows[:batch_size])
            ids = {key: value.to(device) for key, value in batch.items() if key != "records"}
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            try:
                output = model(
                    embedding(ids["context_ids"]),
                    ids["context_mask"],
                    embedding(ids["input_ids"]),
                    ids["input_mask"],
                    embedding(ids["target_ids"]),
                    ids["target_mask"],
                    ids["labels"],
                    ids["qa_context_indices"],
                    loss_only=True,
                )
                loss = output.supervision_losses.mean()
                loss.backward()
                elapsed = time.perf_counter() - start
                peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                print(f"[{dataset_name}] bs={batch_size}: OK loss={float(loss.detach()):.5f} "
                      f"peak={peak:.2f}GiB time={elapsed:.1f}s")
            except torch.cuda.OutOfMemoryError:
                elapsed = time.perf_counter() - start
                peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                print(f"[{dataset_name}] bs={batch_size}: OOM peak={peak:.2f}GiB "
                      f"time={elapsed:.1f}s")
                break
            finally:
                del batch, ids
                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
