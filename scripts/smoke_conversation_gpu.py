"""Minimal real-GPU smoke test for conversation-format QA data.

The test loads the local Qwen checkpoint, renders one QA turn from each requested
JSONL dataset through the tokenizer's native chat template, runs the memory-prefix
QA path, and backpropagates the resulting loss. It is intentionally small: this is
for validating the data/model/device contract, not for measuring training quality.
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

# ``python scripts/<file>.py`` puts ``scripts/`` (rather than the repository
# root) on ``sys.path``. Make direct execution behave like ``python -m``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import make_collate, make_context_dataset
from src.model import load_model
from utils.config import TrainConfig


DATASETS = {
    "c1qa": ("configs/conversation_c1qa.yaml", "ift_c1qa.jsonl"),
    "pwc": ("configs/conversation_pwc.yaml", "ift_pwc.jsonl"),
}


def _configure(path: str, source: Path, samples: int) -> TrainConfig:
    cfg = TrainConfig.from_file(path)
    # Keep the smoke test comfortably within a single 4090 while exercising the
    # same conversation-only collator used by training.
    cfg.data.conversation_file = str(source)
    cfg.data.train_conversation_file = str(source)
    cfg.data.validation_conversation_file = str(source)
    cfg.data.test_conversation_file = str(source)
    cfg.data.max_context_tokens = min(cfg.data.max_context_tokens, 256)
    cfg.data.max_input_tokens = min(cfg.data.max_input_tokens, 512)
    cfg.data.max_question_tokens = min(cfg.data.max_question_tokens, 96)
    cfg.data.max_conversation_tokens = min(cfg.data.max_conversation_tokens, 128)
    cfg.data.max_answer_tokens = min(cfg.data.max_answer_tokens, 32)
    cfg.data.max_samples = samples
    cfg.data.train_max_samples = samples
    # The dataset constructor normally filters source rows by tokenized length. For
    # this smoke test we intentionally exercise truncation on the first row instead
    # of scanning a multi-gigabyte JSONL until a short context is found.
    cfg.data.filter_long_context = False
    cfg.data.train_tasks = "qa"
    cfg.memory.distill_weight = 0.0
    cfg.memory.qa_weight = 1.0
    cfg.memory.recon_weight = 0.0
    cfg.training.batch_size = 1
    cfg.validate()
    return cfg


def _run_one(name: str, cfg: TrainConfig, tokenizer, model, device: torch.device) -> None:
    dataset = make_context_dataset(cfg, "train", tokenizer, limit=cfg.data.max_samples)
    if not len(dataset):
        raise RuntimeError(f"{name}: no usable rows in {dataset.path}")
    collate = make_collate(cfg, tokenizer, sample_qa=True, training=True)
    batch = collate([dataset[0]])
    records = batch.pop("records")
    ids = {key: value.to(device) for key, value in batch.items()}
    embedding = model.qwen.get_input_embeddings()

    # Show the exact rendered prompt once, which also verifies that the native
    # Qwen chat template was used for the structured messages.
    prompt = tokenizer.decode(ids["input_ids"][0], skip_special_tokens=False)
    print(
        f"[{name}] rows={len(dataset)} turns={len(records[0].messages)} "
        f"context_tokens={int(ids['context_mask'][0].sum())} "
        f"question_tokens={int(ids['input_mask'][0].sum())} "
        f"answer_tokens={int(ids['target_mask'][0].sum())}"
    )
    print(f"[{name}] rendered_prompt={prompt!r}")

    model.zero_grad(set_to_none=True)
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
    if output.supervision_losses is None or not torch.isfinite(output.supervision_losses).all():
        raise RuntimeError(f"{name}: invalid QA supervision losses {output.supervision_losses}")
    loss = output.supervision_losses.mean()
    loss.backward()
    gradient_parameters = sum(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    trainable_parameters = sum(parameter.requires_grad for parameter in model.parameters())
    if gradient_parameters == 0:
        raise RuntimeError(f"{name}: backward produced no trainable gradients")
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(
        f"[{name}] loss={float(loss.detach()):.6f} "
        f"gradients={gradient_parameters}/{trainable_parameters} "
        f"peak_cuda_allocated={peak_gib:.2f} GiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/data/lz/contexts")
    parser.add_argument("--model-config", default=None, help="Optional config supplying the model path")
    parser.add_argument("--datasets", default="c1qa,pwc", help="Comma-separated subset of c1qa,pwc")
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this command outside the sandbox on the 4090")
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    names = [item.strip().lower() for item in args.datasets.split(",") if item.strip()]
    unknown = sorted(set(names) - DATASETS.keys())
    if unknown:
        raise ValueError(f"unknown datasets {unknown}; choose from {sorted(DATASETS)}")

    device = torch.device("cuda")
    for index, name in enumerate(names):
        config_path, filename = DATASETS[name]
        if args.model_config:
            config_path = args.model_config
        source = Path(args.data_root) / filename
        if not source.exists():
            raise FileNotFoundError(source)
        cfg = _configure(config_path, source, args.samples)
        tokenizer, model = load_model(cfg)
        model.to(device)
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        print(f"[{name}] model={cfg.model.name_or_path} device={device} dtype={cfg.model.torch_dtype}")
        _run_one(name, cfg, tokenizer, model, device)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        if index + 1 < len(names):
            print()


if __name__ == "__main__":
    main()
