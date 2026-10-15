"""Resuming a run: the model/optimizer/LR come back, and so does the position in the stream.

``--resume`` restores the trainable tensors, the optimizer, the scheduler, the step, the
RNG and the SwanLab run hash. What it used to get wrong was the *data*: the epoch loop restarted
at batch 0 (so a resumed run re-consumed everything it had already seen) and then ran a
full ``epochs`` pass on top of the loaded step, overrunning ``total_steps``. These tests
pin the fixed behaviour, which is what makes a crashed arm salvageable without turning the
A/B into a different experiment.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data import SortishSampler
from utils.checkpoint import CheckpointManager  # noqa: E402

TRAIN_PY = REPO / "scripts" / "train.py"


def _train_module():
    spec = importlib.util.spec_from_file_location("_resume_train_script", TRAIN_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_resume_train_script"] = module
    spec.loader.exec_module(module)
    return module


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 2)


class _CountingDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.reads = []
        self.collated = []

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        self.reads.append(index)
        return index

    def collate(self, indices):
        self.collated.extend(indices)
        return indices


def _counting_loader(monkeypatch, size):
    dataset = _CountingDataset(size)
    monkeypatch.setattr(
        SortishSampler, "_load_or_build_lengths",
        lambda self, *args: list(range(len(self.dataset))),
    )
    sampler = SortishSampler(dataset, None, batch_size=4, bucket_multiplier=2)
    return DataLoader(dataset, batch_size=4, sampler=sampler, collate_fn=dataset.collate), sampler


def test_sampler_offset_skips_dataset_reads_and_collation(monkeypatch):
    loader, sampler = _counting_loader(monkeypatch, 29)
    sampler.set_epoch(2)
    expected = list(loader)[3:]
    loader.dataset.reads.clear()
    loader.dataset.collated.clear()
    sampler.set_start_batch(3)
    assert list(loader) == expected
    assert len(loader) == len(expected)
    wanted = [index for batch in expected for index in batch]
    assert loader.dataset.reads == loader.dataset.collated == wanted
    sampler.set_start_batch(0)
    assert len(loader) == 8


def test_startup_report_lists_trainable_tensors_and_resolves_checkpoint_interval():
    module = _train_module()
    model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))
    model[0].weight.requires_grad_(False)

    parameters, total = module.trainable_parameter_report(model)
    assert [name for name, _, _ in parameters] == ["0.bias", "1.weight", "1.bias"]
    assert total == sum(count for _, _, count in parameters)
    assert module.trainable_parameter_groups(parameters) == [
        ("0", 1, 2),
        ("1", 2, 3),
    ]
    assert module.compact_parameter_count(1_234_567) == "1.2 M"
    table = module.format_trainable_parameter_table(parameters)
    assert "Name" in table and "Type" in table and "Params" in table
    assert "Trainable params" in table and "5" in table
    resampler_table = module.format_trainable_parameter_table(
        [("resampler.latents", (4, 8), 32)]
    )
    assert "question_resampler" in resampler_table
    assert "QuestionResampler" in resampler_table

@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("epoch", (0, 2))
def test_resume_keeps_sharded_tail_and_never_loads_skipped_batches(monkeypatch, rank, epoch):
    from accelerate.data_loader import prepare_data_loader, skip_first_batches

    loader, sampler = _counting_loader(monkeypatch, 53)
    prepared = prepare_data_loader(
        loader, num_processes=4, process_index=rank, device=torch.device("cpu"),
    )
    prepared.set_epoch(epoch)
    expected = list(prepared)[2:]
    loader.dataset.reads.clear()
    loader.dataset.collated.clear()
    sampler.set_epoch(epoch)
    resumed = skip_first_batches(prepared, 2)
    resumed.set_epoch(epoch)
    assert list(resumed) == expected
    assert len(resumed) == len(expected)
    wanted = [index for batch in expected for index in batch]
    assert loader.dataset.reads == loader.dataset.collated == wanted


def test_resume_plan_covers_the_three_cases():
    plan = _train_module().resume_plan
    assert plan(0, 3945, 1) == (0, 0), "fresh run: epoch 0, nothing skipped"
    assert plan(1, 3945, 1) == (0, 1), "resume mid-epoch: start there, skip what was consumed"
    assert plan(2243, 2630, 1) == (0, 2243), "the 14:31 crash point"
    assert plan(2630, 2630, 1) == (1, 0), "exactly at the end: nothing left to do"
    assert plan(3000, 2630, 1) == (1, 0), "past the end stays past the end"
    assert plan(2630, 2630, 2) == (1, 0), "multi-epoch: epoch 2 starts at 0"
    assert plan(3000, 2630, 2) == (1, 370), "multi-epoch mid-way through epoch 2"
    assert plan(5, 0, 1) == (1, 0), "a degenerate loader must not divide by zero"


def test_a_periodic_save_refreshes_last_pt_and_records_the_rank_count():
    with tempfile.TemporaryDirectory() as root:
        model = _Tiny()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        manager = CheckpointManager(root)
        manager.save(model, optimizer, scheduler, 4, {"note": "x"}, world_size=8)

        step_file = Path(root) / "step-4.pt"
        last_file = Path(root) / "last.pt"
        assert step_file.exists() and last_file.exists(), (
            "a periodic save must leave a last.pt as well, or a crash before the final save "
            "leaves nothing to resume from under a stable name"
        )

        restored = _Tiny()
        optimizer2 = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer2, lambda step: 1.0)
        manager2 = CheckpointManager(root)
        step, run_id = manager2.load(last_file, restored, optimizer2, scheduler2)
        assert step == 4
        assert run_id is None
        assert manager2.loaded["world_size"] == 8, "the rank count is needed to warn on resume"
        for name, parameter in model.named_parameters():
            assert torch.equal(parameter, dict(restored.named_parameters())[name]), name


def test_the_final_save_writes_last_pt_too():
    with tempfile.TemporaryDirectory() as root:
        model = _Tiny()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        path = CheckpointManager(root).save(
            model, optimizer, scheduler, 32, {}, final=True, world_size=2,
        )
        assert path.name == "last.pt" and path.exists()
        assert not (Path(root) / "step-32.pt").exists(), "final saves keep the historical name"


def test_a_fresh_run_nests_its_directory_under_the_configured_output_dir():
    """``checkpoint.output_dir`` must actually contain the run's checkpoints.

    It used to be overwritten by ``outputs/<auto run name>``, so the path a config asked for
    never existed: a crashed A/B arm's checkpoints were somewhere in ``outputs/`` under a
    timestamp, and neither the launcher nor a human could find them to resume.
    """
    from utils.config import TrainConfig

    train = _train_module()
    config = TrainConfig.from_file(REPO / "configs/train_baseline.yaml")
    output_root = Path(config.checkpoint.output_dir)
    train._configure_run_paths(config, None, None)
    assert Path(config.checkpoint.output_dir).parent == output_root / "Qwen1.7B"
    assert config.logging.swanlab_run_name in config.checkpoint.output_dir
    assert config.logging.swanlab_log_dir == config.checkpoint.output_dir

    # Resuming keeps the checkpoint's own directory instead of making a new one.
    resumed = TrainConfig.from_file(REPO / "configs/train_baseline.yaml")
    train._configure_run_paths(resumed, None, "outputs/ab_h200_on/Qwen1.7B_x/step-400.pt")
    # Resolve()d on purpose: continuing a run must land in the same absolute directory the
    # checkpoint came from, not in a path relative to wherever the launcher happens to be.
    assert Path(resumed.checkpoint.output_dir).resolve() == Path(
        "outputs/ab_h200_on/Qwen1.7B_x"
    ).resolve()
    assert resumed.logging.swanlab_log_dir == resumed.checkpoint.output_dir


def test_dashboard_name_does_not_replace_timestamped_directory():
    from utils.config import TrainConfig

    train = _train_module()
    config = TrainConfig()
    config.model.name_or_path = "/models/Qwen3-8B"
    config.logging.swanlab_run_name = "reader-off"
    config.logging.swanlab_log_dir = "/old/logs"
    train.new_run_name = lambda _: "Qwen8B_20261015_123456"
    train._configure_run_paths(config, None, None)
    assert config.checkpoint.output_dir == "outputs/Qwen8B/Qwen8B_20261015_123456"
    assert config.logging.swanlab_log_dir == config.checkpoint.output_dir
    assert config.logging.swanlab_run_name == "reader-off"


def test_the_training_loop_honours_the_resumed_position_and_the_step_budget():
    source = TRAIN_PY.read_text(encoding="utf-8")
    compact = "".join(source.split())
    assert "resume_plan(step,steps_per_epoch,cfg.training.epochs)" in compact
    assert "micro_batch_skip(step,steps_per_epoch,accumulation)" in compact
    assert "for epoch in range(start_epoch, cfg.training.epochs)" in source
    # Optimizer steps are what resume plans with; the loop indexes micro-batches.
    assert "skip_first_batches(train_loader, skip_micro_batches)" in source
    assert "sampler.set_start_batch(skip_micro_batches)" in source
    assert "batch_index < skip_micro_batches" not in source
    assert "steps_per_epoch=optimizer_steps(micro_batches_per_rank,accumulation)" in compact
    assert "if step >= total_steps:" in source, (
        "without this a resumed run adds a whole extra epoch on top of the loaded step"
    )
    assert "world_size=" in source, "the rank count must be saved for the resume warning"


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
            except Exception as error:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(error).__name__}: {error}")
            else:
                print(f"ok   {name}")
    print("\nVERDICT:", "ALL PASSED" if not failures else f"{failures} FAILED")
    raise SystemExit(1 if failures else 0)
