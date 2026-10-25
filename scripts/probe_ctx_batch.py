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
from src.data_pipeline import make_collate, make_context_dataset
from src.model import load_model
from utils.checkpoint import CheckpointManager
from utils.machines import machine_names

SYNTHETIC_CONTEXT_TOKENS = 2048
DEFAULT_SYNTHETIC_MAX_BATCH_SIZE = 128


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--ckpt", required=True)
    parser.add_argument("--config")
    parser.add_argument("--machine", choices=machine_names(), default=None)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=128,
        help="Maximum real contexts to load (ignored with --synthetic).",
    )
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--max-batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Probe synthetic contexts with exactly 2048 tokens.",
    )
    return parser.parse_args()


def _load(args):
    cfg = evaluation_config(args.checkpoint, args.config, args.machine)
    if args.synthetic:
        cfg.data.max_context_tokens = SYNTHETIC_CONTEXT_TOKENS
        cfg.data.max_input_tokens = max(
            cfg.data.max_input_tokens, SYNTHETIC_CONTEXT_TOKENS
        )
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
    if args.synthetic:
        return cfg, tokenizer, model, None, None, device
    dataset = make_context_dataset(
        cfg, args.split, tokenizer, limit=args.max_samples, allow_empty=True,
        num_workers=args.num_workers,
    )
    collate = make_collate(cfg, tokenizer, sample_qa=False)
    return cfg, tokenizer, model, dataset, collate, device


def _probe(
    model,
    dataset,
    collate,
    device,
    batch_size,
    num_workers,
    synthetic=False,
):
    saw_batch = False
    peak = 0.0
    max_tokens = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.no_grad():
            if synthetic:
                vocab_size = model.qwen.get_input_embeddings().weight.size(0)
                context_ids = torch.randint(
                    0,
                    vocab_size,
                    (batch_size, SYNTHETIC_CONTEXT_TOKENS),
                    device=device,
                    dtype=torch.long,
                )
                context_mask = torch.ones(
                    (batch_size, SYNTHETIC_CONTEXT_TOKENS),
                    device=device,
                    dtype=torch.bool,
                )
                embedding = model.qwen.get_input_embeddings()
                prefix = model.encode_context_prefix(
                    embedding(context_ids), context_mask
                )
                saw_batch = True
                max_tokens = SYNTHETIC_CONTEXT_TOKENS
                del prefix, context_ids, context_mask
            else:
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collate,
                    num_workers=num_workers,
                )
                for batch in loader:
                    context_ids = batch["context_ids"].to(device, non_blocking=True)
                    context_mask = batch["context_mask"].to(device, non_blocking=True)
                    embedding = model.qwen.get_input_embeddings()
                    model.encode_context_prefix(embedding(context_ids), context_mask)
                    saw_batch = True
                    max_tokens = max(max_tokens, int(context_ids.size(1)))
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                        peak = max(peak, torch.cuda.max_memory_allocated(device) / 2**30)
                    del context_ids, context_mask, batch
                del loader
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak = max(peak, torch.cuda.max_memory_allocated(device) / 2**30)
        return (saw_batch, None, peak, max_tokens)
    except torch.cuda.OutOfMemoryError as error:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return False, str(error).splitlines()[0], peak, max_tokens
    finally:
        gc.collect()


def main():
    args = _parse_args()
    if args.min_batch_size < 1 or args.max_batch_size < 0:
        raise SystemExit("batch sizes and max-samples must be positive")
    if not args.synthetic and args.max_samples < 1:
        raise SystemExit("batch sizes and max-samples must be positive")
    if args.max_batch_size and args.max_batch_size < args.min_batch_size:
        raise SystemExit("max-batch-size must be >= min-batch-size")
    cfg, _tokenizer, model, dataset, collate, device = _load(args)
    if device.type != "cuda":
        print("CUDA is unavailable; running the probe on CPU has no OOM search value.")
        return
    if args.synthetic:
        upper = args.max_batch_size or max(
            args.min_batch_size, DEFAULT_SYNTHETIC_MAX_BATCH_SIZE
        )
    else:
        upper = args.max_batch_size or max(args.min_batch_size, min(len(dataset), 128))
        upper = min(upper, len(dataset))
    low = args.min_batch_size - 1
    high = upper + 1
    while low + 1 < high:
        candidate = (low + high) // 2
        torch.cuda.empty_cache()
        ok, error, peak, max_tokens = _probe(
            model,
            dataset,
            collate,
            device,
            candidate,
            args.num_workers,
            synthetic=args.synthetic,
        )
        if ok:
            low = candidate
            context_label = (
                f"synthetic_context_tokens={SYNTHETIC_CONTEXT_TOKENS}"
                if args.synthetic
                else f"padded_context_tokens={max_tokens}"
            )
            print(
                f"ctx_batch_size={candidate}: OK, peak_allocated={peak:.2f} GiB, "
                f"{context_label}"
            )
        else:
            high = candidate
            print(f"ctx_batch_size={candidate}: OOM ({error})")
    if low < args.min_batch_size:
        raise SystemExit("minimum ctx_batch_size does not fit in available GPU memory")
    print(f"recommended ctx_batch_size: {low}")
    tested_tokens = (
        SYNTHETIC_CONTEXT_TOKENS if args.synthetic else cfg.data.max_context_tokens
    )
    print(f"tested max_context_tokens: {tested_tokens}")
    print("Use a smaller value for headroom when context lengths vary across datasets.")


if __name__ == "__main__":
    main()
