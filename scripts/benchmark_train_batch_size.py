"""Find the largest per-GPU training batch that fits for conversation QA."""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

# ``python scripts/<file>.py`` puts ``scripts/`` (rather than the repository
# root) on ``sys.path``. Make direct execution behave like ``python -m``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import ContextRecord, ConversationMessage
from src.data_pipeline import make_collate, make_context_dataset
from src.model import load_model
from utils.config import TrainConfig

DATASETS = {
    "c1qa": ("configs/conversation_c1qa.yaml", "ift_c1qa.jsonl"),
    "pwc": ("configs/conversation_pwc.yaml", "ift_pwc.jsonl"),
}


def configure(
    config_path: str,
    source: Path,
    rows: int,
    max_context_tokens: int,
    max_question_tokens: int,
    max_answer_tokens: int,
    machine: str,
) -> TrainConfig:
    cfg = TrainConfig.from_file(config_path, machine=machine)
    for name in ("train_conversation_file",
                 "validation_conversation_file", "test_conversation_file"):
        setattr(cfg.data, name, str(source))
    cfg.data.max_samples = rows
    cfg.data.train_max_samples = rows
    # Explicit stress-test budget: context / question / answer.
    cfg.data.max_context_tokens = max_context_tokens
    cfg.data.max_question_tokens = max_question_tokens
    cfg.data.max_answer_tokens = max_answer_tokens
    cfg.data.train_stage = "only_qa"
    cfg.loss.qa_weight = 1.0
    cfg.loss.recon_weight = 0.0
    # Read the first rows and let the normal collator truncate them to the
    # configured training limits instead of scanning for short examples.
    cfg.data.filter_long_context = False
    cfg.validate()
    return cfg


def extreme_records(tokenizer, count: int, context_tokens: int,
                    question_tokens: int, answer_tokens: int):
    """Build records that are at least as long as each configured limit.

    The collator truncates these fields to the requested limits. Using repeated
    ordinary text is more robust across fast tokenizer implementations than
    trying to round-trip arbitrary vocabulary ids through ``decode``.
    """
    def long_text(length: int, piece: str) -> str:
        return (piece + " ") * max(length, 1)

    context = long_text(context_tokens * 2, "context_token")
    question = long_text(question_tokens * 2, "question_token")
    answer = long_text(answer_tokens * 2, "answer_token")
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
    parser.add_argument("--datasets", default="c1qa")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument(
        "--sample-mode",
        choices=("extreme", "source"),
        default="extreme",
        help="Use max-length synthetic records (default) or source JSONL records",
    )
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--machine", default="4090")
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--max-question-tokens", type=int, default=96)
    parser.add_argument("--max-answer-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--worker-batch-size", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run outside the sandbox on a real 4090")
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    if not batch_sizes or any(value <= 0 for value in batch_sizes):
        raise ValueError("--batch-sizes must contain positive integers")
    names = [value.strip().lower() for value in args.datasets.split(",") if value.strip()]
    unknown = sorted(set(names) - DATASETS.keys())
    if unknown:
        parser.error(f"unknown datasets: {', '.join(unknown)}")
    if args.worker_batch_size is None:
        all_results = []
        for dataset_name in names:
            for batch_size in batch_sizes:
                worker_output = Path("/tmp") / (
                    f"memoryqwen-bs-probe-{os.getpid()}-{dataset_name}-{batch_size}.json"
                )
                command = [
                    sys.executable, str(Path(__file__).resolve()),
                    "--data-root", args.data_root,
                    "--datasets", dataset_name,
                    "--batch-sizes", str(batch_size),
                    "--rows", str(max(args.rows, batch_size)),
                    "--sample-mode", args.sample_mode,
                    "--model-config", args.model_config or "",
                    "--machine", args.machine,
                    "--max-context-tokens", str(args.max_context_tokens),
                    "--max-question-tokens", str(args.max_question_tokens),
                    "--max-answer-tokens", str(args.max_answer_tokens),
                    "--worker-batch-size", str(batch_size),
                    "--worker-output", str(worker_output),
                ]
                command = [value for value in command if value != ""]
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
                completed = subprocess.run(command, env=env, check=False)
                if worker_output.exists():
                    try:
                        all_results.extend(json.loads(worker_output.read_text()).get("results", []))
                    finally:
                        worker_output.unlink(missing_ok=True)
                elif completed.returncode != 0:
                    all_results.append({
                        "dataset": dataset_name,
                        "batch_size": batch_size,
                        "status": "failed",
                        "return_code": completed.returncode,
                    })
        if args.output is not None:
            successful = [
                row["batch_size"] for row in all_results
                if row.get("status") == "ok"
            ]
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({
                "model_config": args.model_config,
                "machine": args.machine,
                "datasets": names,
                "batch_sizes": batch_sizes,
                "rows": args.rows,
                "max_context_tokens": args.max_context_tokens,
                "max_question_tokens": args.max_question_tokens,
                "max_answer_tokens": args.max_answer_tokens,
                "results": all_results,
                "maximum_successful_batch_size": max(successful) if successful else None,
            }, indent=2) + "\n")
            print(f"Saved batch-size results to {args.output}")
        return

    batch_sizes = [args.worker_batch_size]
    device = torch.device("cuda")

    results = []
    for dataset_name in names:
        config_path, filename = DATASETS[dataset_name]
        if args.model_config:
            config_path = args.model_config
        cfg = configure(
            config_path,
            Path(args.data_root) / filename,
            args.rows,
            args.max_context_tokens,
            args.max_question_tokens,
            args.max_answer_tokens,
            args.machine,
        )
        tokenizer, model = load_model(cfg)
        model.to(device).train()
        embedding = model.qwen.get_input_embeddings()
        collate = make_collate(cfg, tokenizer, sample_qa=True, training=True)
        if args.sample_mode == "extreme":
            rows = list(extreme_records(
                tokenizer,
                args.rows,
                args.max_context_tokens,
                args.max_question_tokens,
                args.max_answer_tokens,
            ))
        else:
            dataset = make_context_dataset(cfg, "train", tokenizer, limit=args.rows)
            rows = [dataset[index] for index in range(len(dataset))]
        print(f"[{dataset_name}] model={cfg.model.name_or_path}")
        print(f"[{dataset_name}] sample_mode={args.sample_mode} rows={len(rows)}")
        print(f"[{dataset_name}] limits=context={cfg.data.max_context_tokens}, "
              f"question={cfg.data.max_question_tokens}, answer={cfg.data.max_answer_tokens}")
        dataset_results = []
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
                    embedding(ids["qa_input_ids"]),
                    ids["qa_input_mask"],
                    embedding(ids["qa_target_ids"]),
                    ids["qa_target_mask"],
                    ids["qa_labels"],
                    ids["qa_context_indices"],
                    question_prefix_embeds=embedding(ids["qa_prefix_ids"]),
                    question_prefix_mask=ids["qa_prefix_mask"],
                    answer_prefix_embeds=embedding(ids["qa_answer_prefix_ids"]),
                    answer_prefix_mask=ids["qa_answer_prefix_mask"],
                    question_readout_mask=ids["qa_readout_mask"],
                    loss_only=True,
                )
                loss = output.supervision_losses.mean()
                loss.backward()
                elapsed = time.perf_counter() - start
                peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                print(f"[{dataset_name}] bs={batch_size}: OK loss={float(loss.detach()):.5f} "
                      f"peak={peak:.2f}GiB time={elapsed:.1f}s")
                result = {
                    "dataset": dataset_name,
                    "batch_size": batch_size,
                    "status": "ok",
                    "loss": float(loss.detach()),
                    "peak_memory_gib": peak,
                    "seconds": elapsed,
                }
                results.append(result)
                dataset_results.append(result)
            except torch.cuda.OutOfMemoryError:
                elapsed = time.perf_counter() - start
                peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                print(f"[{dataset_name}] bs={batch_size}: OOM peak={peak:.2f}GiB "
                      f"time={elapsed:.1f}s")
                result = {
                    "dataset": dataset_name,
                    "batch_size": batch_size,
                    "status": "oom",
                    "peak_memory_gib": peak,
                    "seconds": elapsed,
                }
                results.append(result)
                dataset_results.append(result)
                break
            finally:
                del batch, ids
                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        successful = [
            row["batch_size"] for row in dataset_results if row["status"] == "ok"
        ]
        print(
            f"[{dataset_name}] recommended maximum batch size: "
            f"{max(successful) if successful else 'none'}"
        )
    output_path = args.worker_output or args.output
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({
            "model_config": args.model_config,
            "machine": args.machine,
            "datasets": names,
            "batch_sizes": batch_sizes,
            "rows": args.rows,
            "max_context_tokens": args.max_context_tokens,
            "max_question_tokens": args.max_question_tokens,
            "max_answer_tokens": args.max_answer_tokens,
            "results": results,
        }, indent=2) + "\n")
        print(f"Saved batch-size results to {output_path}")


if __name__ == "__main__":
    main()
