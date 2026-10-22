"""The teacher-forced reconstruction metric: one forward, no decoding.

Recon is the objective that says whether the memory actually carries the context, so the
training loop should be able to watch it. It used to be scored by generating the whole context
(``max_context_tokens`` greedy steps per context), which is why the loop skipped it entirely;
the metric here is the training objective's own teacher-forced next-token prediction, so it
costs one forward and is exactly the number the loss optimises.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.evaluator import Evaluator
from src.model import ContextMemoryModel
from utils.config import TrainConfig

VOCAB = 64


def _model() -> ContextMemoryModel:
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=256, tie_word_embeddings=True,
    )
    config._attn_implementation = "sdpa"
    model = ContextMemoryModel(
        Qwen3ForCausalLM(config), rank=2, memory_length=4, target_modules=["q_proj"],
        memory_resampler_hidden_size=16, memory_resampler_heads=2,
        recon_user_token_ids=(2,), recon_answer_prefix_token_ids=(3, 4),
    )
    return model.eval()


class _Loader:
    """The three batch fields the reconstruction pass reads."""

    def __init__(self, batches):
        self.batches = list(batches)
        self.batch_size = self.batches[0]["context_ids"].size(0)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def _batch(datasets=("squad", "race"), length: int = 6):
    torch.manual_seed(3)
    contexts = len(datasets)
    ids = torch.randint(2, VOCAB, (contexts, length))
    return {
        "context_ids": ids,
        "context_mask": torch.ones(contexts, length, dtype=torch.bool),
        "qa_context_indices": torch.arange(contexts),
        "records": [SimpleNamespace(dataset=name) for name in datasets],
    }


def test_the_metric_is_the_training_objective_scored_token_by_token():
    model = _model()
    batch = _batch()
    metrics = Evaluator(None, TrainConfig()).recon_teacher_forced(
        model, _Loader([batch]), torch.device("cpu")
    )

    ids, mask = batch["context_ids"], batch["context_mask"]
    labels = ids.masked_fill(~mask, -100)
    embedding = model.qwen.get_input_embeddings()
    prefix = model.encode_context_prefix(embedding(ids), mask)
    output = model.forward_recon_with_memory_prefix(
        prefix, torch.arange(ids.size(0)), embedding(ids), mask, labels,
        loss_only=True, return_stats=True,
    )
    stats = output.row_stats
    losses = output.supervision_losses

    # Per row, the training loss *is* nll / tokens of the metric ...
    torch.testing.assert_close(losses, stats["nll"] / stats["tokens"].clamp_min(1))
    # ... and the reported cross entropy is that over the whole evaluation.
    torch.testing.assert_close(
        torch.tensor(metrics["recon_ce"]),
        torch.tensor(float(stats["nll"].sum() / stats["tokens"].sum())),
    )
    assert metrics["contexts"] == 2


def test_the_metric_reports_a_usable_accuracy_and_per_dataset_keys():
    model = _model()
    metrics = Evaluator(None, TrainConfig()).recon_teacher_forced(
        model, _Loader([_batch()]), torch.device("cpu")
    )
    assert 0.0 <= metrics["recon_token_accuracy"] <= 1.0
    assert metrics["recon_ce"] > 0.0
    assert metrics["recon_ppl"] > 1.0
    for dataset in ("squad", "race"):
        assert 0.0 <= metrics[f"dataset/{dataset}/recon_token_accuracy"] <= 1.0
        assert metrics[f"dataset/{dataset}/recon_ce"] > 0.0
        assert metrics[f"dataset/{dataset}/contexts"] == 1


def test_recon_evaluation_projects_each_scored_chunk_once(monkeypatch):
    import torch.nn.functional as functional

    model = _model()
    batch = _batch()
    ids, mask = batch["context_ids"], batch["context_mask"]
    embedding = model.qwen.get_input_embeddings()
    prefix = model.encode_context_prefix(embedding(ids), mask)
    head_weight = model.qwen.get_output_embeddings().weight
    calls = 0
    original = functional.linear

    def counted_linear(input, weight, bias=None):
        nonlocal calls
        calls += weight is head_weight
        return original(input, weight, bias)

    monkeypatch.setattr("src.losses.F.linear", counted_linear)
    model.recon_teacher_forced(
        prefix, embedding(ids), mask, ids.masked_fill(~mask, -100),
    )
    assert calls == 1


def test_all_supplied_contexts_are_scored_without_a_separate_recon_budget():
    model = _model()
    loader = _Loader([_batch(("squad", "race")), _batch(("squad", "race"))])
    metrics = Evaluator(None, TrainConfig()).recon_teacher_forced(
        model, loader, torch.device("cpu")
    )
    assert metrics["contexts"] == 4
