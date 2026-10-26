from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.model import ContextMemoryOutput  # noqa: E402
from src.training.distributed import ddp_kwargs_for_stage  # noqa: E402
from src.training.loop import run_training_loop  # noqa: E402
from utils.config import TrainConfig  # noqa: E402


class _Sampler(Sampler[int]):
    def __init__(self):
        self.epoch = 0

    def __iter__(self):
        return iter(range(2))

    def __len__(self):
        return 2

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_start_batch(self, batch):
        pass


class _Dataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        value = index + 1
        return {
            "context_ids": torch.tensor([value, value + 1]),
            "context_mask": torch.ones(2, dtype=torch.bool),
            "qa_input_ids": torch.tensor([[value, value + 1]]),
            "qa_input_mask": torch.ones(1, 2, dtype=torch.bool),
            "qa_target_ids": torch.tensor([[value + 1, value + 2]]),
            "qa_target_mask": torch.ones(1, 2, dtype=torch.bool),
            "qa_labels": torch.tensor([[value + 1, value + 2]]),
            "qa_context_indices": torch.tensor([0]),
            "qa_prefix_ids": torch.tensor([[value]]),
            "qa_prefix_mask": torch.ones(1, 1, dtype=torch.bool),
            "qa_readout_mask": torch.ones(1, 2, dtype=torch.bool),
            "qa_answer_prefix_ids": torch.tensor([[value]]),
            "qa_answer_prefix_mask": torch.ones(1, 1, dtype=torch.bool),
            "recon_target_ids": torch.tensor([[value + 2, value + 3]]),
            "recon_target_mask": torch.ones(1, 2, dtype=torch.bool),
            "recon_labels": torch.tensor([[value + 2, value + 3]]),
            "recon_context_indices": torch.tensor([0]),
        }


class _StageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.embedding.requires_grad_(False)
        self.qwen = SimpleNamespace(get_input_embeddings=lambda: self.embedding)
        self.qa_weight = nn.Parameter(torch.tensor(0.7))
        self.recon_weight = nn.Parameter(torch.tensor(0.9))
        self.shared = nn.Parameter(torch.tensor(0.2))
        self.context_layer_logits = nn.Parameter(torch.zeros(1))
        self.context_layer_indices = [-1]
        self.recon_task_embedding = nn.Parameter(torch.zeros(4))
        self.context_resampler = nn.Linear(1, 1)

    def forward(
        self,
        context_embeds,
        context_mask,
        qa_input_embeds,
        qa_input_mask,
        qa_target_embeds,
        qa_target_mask,
        qa_labels,
        qa_context_indices,
        *,
        recon_batch=None,
        **kwargs,
    ):
        losses = []
        if qa_input_embeds.size(0):
            qa_signal = qa_input_embeds.flatten(1).mean(dim=1)
            losses.append((qa_signal * self.qa_weight + self.shared).square())
        if recon_batch is not None:
            recon_signal = recon_batch["context_embeds"].flatten(1).mean(dim=1)
            losses.append((recon_signal * self.recon_weight + self.shared).square())
        supervision_losses = torch.cat(losses)
        return ContextMemoryOutput(
            logits=None,
            labels=qa_labels,
            memory=context_embeds,
            supervision_losses=supervision_losses,
            supervision_mask=torch.ones_like(supervision_losses, dtype=torch.bool),
        )


class _Collate:
    def set_epoch(self, epoch):
        self.epoch = epoch


class _CheckpointManager:
    best_f1 = None
    best_step = 0

    def save(self, *args, **kwargs):
        return None

    def update_best(self, *args):
        return False


def _run_two_stage_worker():
    from accelerate import Accelerator

    assert int(os.environ["WORLD_SIZE"]) == 2
    accelerator = Accelerator(
        mixed_precision="no",
        kwargs_handlers=ddp_kwargs_for_stage("two_stage"),
    )
    cfg = TrainConfig.from_dict(
        {
            "training": {"epochs": 1, "batch_size": 1, "max_grad_norm": 10.0},
            "data": {"train_stage": "two_stage"},
            "evaluation": {"autoregressive_every": 1000},
            "logging": {"log_every": 1000},
        }
    )
    cfg.validate()
    model = _StageModel().to(accelerator.device)
    initial = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if name in {"qa_weight", "recon_weight"}
    }
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    sampler = _Sampler()
    loader = DataLoader(_Dataset(), batch_size=1, sampler=sampler)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    manager = _CheckpointManager()
    managers = {"recon": manager, "qa": manager}
    schedulers = {
        stage: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        for stage in managers
    }
    step, _ = run_training_loop(
        cfg,
        model,
        optimizer,
        schedulers,
        loader,
        [],
        sampler,
        _Collate(),
        None,
        None,
        None,
        None,
        None,
        managers,
        manager,
        0,
        0,
        0,
        2,
        2,
        1,
        1,
        1,
        accelerator,
        accelerator.device,
        None,
        [],
    )
    assert step == 2
    trained = accelerator.unwrap_model(model)
    assert not torch.equal(initial["qa_weight"], trained.qa_weight)
    assert not torch.equal(initial["recon_weight"], trained.recon_weight)
    for parameter in trained.parameters():
        replicas = [torch.empty_like(parameter) for _ in range(2)]
        dist.all_gather(replicas, parameter.detach())
        torch.testing.assert_close(replicas[0], replicas[1])
    accelerator.wait_for_everyone()
    accelerator.end_training()


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least two CUDA GPUs")
def test_two_stage_training_updates_both_phases_on_two_ddp_ranks():
    env = os.environ.copy()
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        env["CUDA_VISIBLE_DEVICES"] = "0,1"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        str(Path(__file__).resolve()),
        "--two-stage-worker",
    ]
    subprocess.run(command, cwd=REPO, env=env, check=True, timeout=180)


if __name__ == "__main__" and "--two-stage-worker" in sys.argv:
    _run_two_stage_worker()
