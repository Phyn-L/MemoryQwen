from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from src.training.loop import run_training_loop
from utils.config import TrainConfig


class _OneStepModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 4)
        self.qwen = SimpleNamespace(get_input_embeddings=lambda: self.embed)
        self.context_resampler = SimpleNamespace()
        self.context_layer_logits = nn.Parameter(torch.zeros(1))
        self.context_layer_indices = [-1]
        self.recon_task_embedding = nn.Parameter(torch.zeros(4))

    def forward(self, *args, **kwargs):
        loss = self.context_layer_logits.sum() + self.recon_task_embedding.sum() * 0
        return SimpleNamespace(
            supervision_losses=torch.stack([loss, loss]),
            supervision_mask=torch.ones(2, dtype=torch.bool),
        )


class _Loader:
    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        return iter([self.batch])

    def __len__(self):
        return 1


class _Sampler:
    def set_epoch(self, epoch):
        pass

    def set_start_batch(self, batch):
        pass


class _Collate:
    def set_epoch(self, epoch):
        pass


class _Manager:
    best_f1 = None
    best_step = 0

    def save(self, *args, **kwargs):
        return None

    def update_best(self, *args):
        return False


def test_training_loop_terminates_after_last_batch_without_finished_state():
    cfg = TrainConfig.from_dict(
        {
            "training": {"epochs": 1, "batch_size": 1},
            "data": {"train_stage": "only_qa"},
            "evaluation": {"autoregressive_every": 1000},
            "logging": {"log_every": 1000},
        }
    )
    cfg.validate()
    long_ids = {
        "context_ids": (1, 2),
        "context_mask": (1, 2),
        "qa_input_ids": (1, 2),
        "qa_input_mask": (1, 2),
        "qa_target_ids": (1, 2),
        "qa_target_mask": (1, 2),
        "qa_labels": (1, 2),
        "qa_context_indices": (1,),
        "qa_prefix_ids": (1, 1),
        "qa_prefix_mask": (1, 1),
        "qa_readout_mask": (1, 2),
        "qa_answer_prefix_ids": (1, 1),
        "qa_answer_prefix_mask": (1, 1),
        "recon_target_ids": (1, 2),
        "recon_target_mask": (1, 2),
        "recon_labels": (1, 2),
        "recon_context_indices": (1,),
    }
    batch = {
        key: torch.ones(shape, dtype=torch.bool if "mask" in key else torch.long)
        for key, shape in long_ids.items()
    }
    model = _OneStepModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    manager = _Manager()
    step, _ = run_training_loop(
        cfg,
        model,
        optimizer,
        scheduler,
        _Loader(batch),
        [],
        _Sampler(),
        _Collate(),
        SimpleNamespace(),
        None,
        None,
        None,
        None,
        {"only_qa": manager},
        manager,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        accumulation=1,
        accelerator=None,
        device=torch.device("cpu"),
        run=None,
        trainable=[],
    )
    assert step == 1
