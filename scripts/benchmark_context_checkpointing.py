"""Compare native Transformers layer checkpointing with PyTorch contiguous segments.

The segmented implementation is a benchmark reference only; training uses Transformers.
Reported times cover forward/backward, excluding optimizer updates and data loading.
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import transformers
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.model import load_model
from utils.config import TrainConfig


class _LayerSegment(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def _forward(self, hidden_states, **kwargs):
        for layer in self.layers:
            hidden_states = layer(hidden_states, **kwargs)
        return hidden_states

    def forward(self, hidden_states, **kwargs):
        return checkpoint(
            self._forward, hidden_states, use_reentrant=False,
            preserve_rng_state=True, **kwargs,
        )


@contextmanager
def reference_segments(model, interval):
    """Keep the experimental layer replacement out of the training implementation."""
    body = model._transformer_body
    if len(set(body.config.layer_types)) > 1:
        raise ValueError("the segmented benchmark requires homogeneous attention types")
    original_encode = model._encode_context_hidden

    def encode(*args):
        layers = body.layers
        enabled = model.context_gradient_checkpointing
        try:
            model.context_gradient_checkpointing = False
            body.layers = nn.ModuleList(
                _LayerSegment(layers[start:start + interval])
                for start in range(0, len(layers), interval)
            )
            return original_encode(*args)
        finally:
            body.layers = layers
            model.context_gradient_checkpointing = enabled

    model._encode_context_hidden = encode
    try:
        yield
    finally:
        model._encode_context_hidden = original_encode


def gradient_differences(model, reference):
    """Report numerical errors, without claiming bitwise BF16 CUDA equivalence."""
    current = {
        name: p.grad.detach().cpu().clone() if p.grad is not None else None
        for name, p in model.named_parameters() if p.requires_grad
    }
    if reference is None:
        return current, {"gradient_reference": True, "trainable_tensors": len(current)}
    assert current.keys() == reference.keys()
    errors = []
    compared = 0
    for name, actual in current.items():
        expected = reference[name]
        if (actual is None) != (expected is None):
            raise AssertionError(f"gradient presence differs: {name}")
        if actual is None:
            continue
        compared += 1
        delta = actual.float() - expected.float()
        relative_l2 = float(delta.norm() / expected.float().norm().clamp_min(1e-12))
        errors.append((relative_l2, name, float(delta.abs().max())))
    return reference, {
        "gradient_compared_tensors": compared,
        "worst_gradient_errors_relative_l2_name_max_abs": sorted(errors, reverse=True)[:3],
    }


def run_case(model, args, backend, interval, reference):
    model.gradient_checkpoint_every_n_layers = interval
    model.train()
    device = torch.device(args.device)
    torch.manual_seed(123)
    vocab = model.qwen.config.vocab_size
    ids = torch.randint(vocab, (args.batch_size, args.sequence_length), device=device)
    question = torch.randint(vocab, (args.batch_size, 32), device=device)
    answer = torch.randint(vocab, (args.batch_size, 32), device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)
    question_mask = torch.ones_like(question, dtype=torch.bool)
    answer_mask = torch.ones_like(answer, dtype=torch.bool)
    indices = torch.arange(args.batch_size, device=device)
    embedding = model.qwen.get_input_embeddings()

    def step():
        model.zero_grad(set_to_none=True)
        torch.manual_seed(42)
        prefix = model.encode_context_prefix(embedding(ids), mask)
        if args.scope == "writer":
            loss = prefix.memory.float().square().mean()
        else:
            qa = model.forward_with_memory_prefix(
                prefix, indices, embedding(question), question_mask,
                embedding(answer), answer_mask, answer, loss_only=True,
            )
            recon = model.forward_recon_with_memory_prefix(
                prefix, indices, embedding(ids), mask, ids, loss_only=True,
            )
            loss = qa.supervision_losses.mean() + recon.supervision_losses.mean()
        loss.backward()
        return float(loss.detach())

    result = {"backend": backend, "interval": interval}
    context = reference_segments(model, interval) if backend == "pytorch" else nullcontext()
    try:
        with context:
            loss = step()
            reference, differences = gradient_differences(model, reference)
            result.update(differences)
            result["loss"] = loss
            for _ in range(args.warmup):
                step()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            timings = []
            for _ in range(args.iterations):
                started = time.perf_counter()
                step()
                torch.cuda.synchronize(device)
                timings.append(time.perf_counter() - started)
            seconds = statistics.median(timings)
            result.update(
                peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024**3,
                step_seconds_median=seconds,
                steps_per_second=1 / seconds,
            )
    except torch.cuda.OutOfMemoryError:
        result["oom"] = True
    finally:
        model.zero_grad(set_to_none=True)
    return result, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_baseline.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=1536)
    parser.add_argument("--intervals", type=int, nargs="+", default=[1, 2, 4, 6])
    parser.add_argument("--scope", choices=["writer", "joint"], default="joint")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Optional JSONL report path")
    args = parser.parse_args()
    if min(args.batch_size, args.sequence_length, args.iterations, *args.intervals) < 1:
        parser.error("batch size, sequence length, iterations and intervals must be positive")
    if args.warmup < 0:
        parser.error("warmup must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    cfg = TrainConfig.from_file(args.config)
    cfg.model.context_gradient_checkpointing = True
    cfg.training.gradient_checkpoint_every_n_layers = 1
    torch.manual_seed(42)
    _, model = load_model(cfg)
    model.to(device)
    if args.sequence_length > model.max_context_tokens:
        raise ValueError("sequence length exceeds configured max_context_tokens")
    metadata = {
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "transformers": transformers.__version__, "model": cfg.model.name_or_path,
        "dtype": str(model.dtype), "layers": model.qwen.config.num_hidden_layers,
        "batch_size": args.batch_size, "sequence_length": args.sequence_length,
        "memory_length": cfg.memory.memory_length, "context_layer_indices": model.context_layer_indices,
        "scope": args.scope, "question_length": 32, "answer_length": 32,
        "iterations": args.iterations, "warmup": args.warmup,
        "excludes": "optimizer updates, data loading, gradient synchronization",
        "gradient_reference": "Transformers interval=1; BF16 CUDA errors reported numerically",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") if args.output else nullcontext() as output:
        def emit(record):
            line = json.dumps(record)
            print(line, flush=True)
            if output:
                output.write(line + "\n")
                output.flush()

        emit(metadata)
        reference = None
        cases = [("transformers", 1)] + [
            (backend, interval) for backend in ("transformers", "pytorch")
            for interval in args.intervals if (backend, interval) != ("transformers", 1)
        ]
        for backend, interval in cases:
            gc.collect()
            torch.cuda.empty_cache()
            result, reference = run_case(model, args, backend, interval, reference)
            emit(result)


if __name__ == "__main__":
    main()
