"""Find the largest evaluation context batch that fits GPU memory."""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from scripts.test import evaluation_config
from src.model import load_model
from src.pipeline import make_collate, make_context_dataset
from utils.checkpoint import CheckpointManager
from utils.machines import machine_names


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--ckpt", required=True)
    parser.add_argument("--config")
    parser.add_argument("--machine", choices=machine_names(), default=None)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--max-batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _load(args):
    cfg = evaluation_config(args.checkpoint, args.config, args.machine)
    cfg.data.filter_qa_pair_answers = True
    cfg.data.max_answer_tokens = cfg.evaluation.max_new_tokens
    cfg.data.eval_num_workers = args.num_workers
    cfg.evaluation.ctx_batch_size = args.min_batch_size
    cfg.validate()
    tokenizer, model = load_model(cfg)
    checkpoint = Path(args.checkpoint).resolve()
    cfg.checkpoint.output_dir = str(checkpoint.parent)
    CheckpointManager(cfg.checkpoint.output_dir).load(checkpoint, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dataset = make_context_dataset(
        cfg, args.split, tokenizer, limit=args.max_samples, allow_empty=True,
        num_workers=args.num_workers,
    )
    collate = make_collate(cfg, tokenizer, sample_qa=False)
    return cfg, tokenizer, model, dataset, collate, device


def _probe(model, dataset, collate, device, batch_size, num_workers):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate,
        num_workers=num_workers,
    )
    saw_batch = False
    try:
        with torch.no_grad():
            for batch in loader:
                context_ids = batch["context_ids"].to(device, non_blocking=True)
                context_mask = batch["context_mask"].to(device, non_blocking=True)
                embedding = model.qwen.get_input_embeddings()
                model.encode_context_prefix(embedding(context_ids), context_mask)
                saw_batch = True
                del context_ids, context_mask, batch
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
        return saw_batch, None
    except torch.cuda.OutOfMemoryError as error:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return False, str(error).splitlines()[0]
    finally:
        del loader
        gc.collect()


def main():
    args = _parse_args()
    if args.min_batch_size < 1 or args.max_batch_size < 0 or args.max_samples < 1:
        raise SystemExit("batch sizes and max-samples must be positive")
    cfg, _tokenizer, model, dataset, collate, device = _load(args)
    if device.type != "cuda":
        print("CUDA is unavailable; running the probe on CPU has no OOM search value.")
        return
    upper = args.max_batch_size or max(args.min_batch_size, min(len(dataset), 128))
    upper = min(upper, len(dataset))
    low = args.min_batch_size - 1
    high = upper + 1
    while low + 1 < high:
        candidate = (low + high) // 2
        torch.cuda.empty_cache()
        ok, error = _probe(model, dataset, collate, device, candidate, args.num_workers)
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        if ok:
            low = candidate
            print(f"ctx_batch_size={candidate}: OK, peak_allocated={peak:.2f} GiB")
        else:
            high = candidate
            print(f"ctx_batch_size={candidate}: OOM ({error})")
    if low < args.min_batch_size:
        raise SystemExit("minimum ctx_batch_size does not fit in available GPU memory")
    print(f"recommended ctx_batch_size: {low}")
    print("Use a smaller value for headroom when context lengths vary across datasets.")


if __name__ == "__main__":
    main()
