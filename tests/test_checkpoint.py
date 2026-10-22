"""Checkpoint selection, retention and alias integrity on CPU."""
import torch

from utils.checkpoint import CheckpointManager
from utils.config import TrainConfig


def test_best_retention_resume_and_final_alias_integrity(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    manager = CheckpointManager(tmp_path)

    def save(step, score=None, final=False):
        with torch.no_grad():
            model.weight.fill_(step)
        manager.save(model, optimizer, scheduler, step, {},
                     validation_f1=score, final=final)

    def read(name):
        return torch.load(tmp_path / name, weights_only=False)

    for step, score in [(2, .8), (10, .4), (20, .8), (30, float('nan'))]:
        save(step, score)
    assert {p.name for p in tmp_path.glob('step-*.pt')} == {
        'step-10.pt', 'step-20.pt', 'step-30.pt'}
    assert read('best.pt')['step'] == 2
    assert read('last.pt')['step'] == 30
    manager = CheckpointManager(tmp_path)
    manager.load(tmp_path / 'last.pt', model, optimizer, scheduler)
    assert manager.best_f1 == .8 and manager.best_step == 2
    save(40, .7)
    assert read('best.pt')['step'] == 2
    save(50, .9)
    assert read('best.pt')['step'] == 50
    save(51, final=True)
    assert read('last.pt')['step'] == 51
    for name in ('best.pt', 'step-50.pt'):
        state = read(name)
        assert state['step'] == 50
        assert state['model']['weight'].item() == 50
    assert read('last.pt')['best_f1'] == .9
    assert len(list(tmp_path.glob('*.pt'))) == 5


def test_pre_eval_checkpoint_can_be_selected_as_best_after_eval(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    manager = CheckpointManager(tmp_path)
    with torch.no_grad():
        model.weight.fill_(12)
    checkpoint = manager.save(model, optimizer, scheduler, 12, {})

    assert manager.update_best(checkpoint, .75)
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    assert best["step"] == 12
    assert best["validation_f1"] == .75
    assert best["best_f1"] == .75
    assert not manager.update_best(checkpoint, .7)

    resumed = CheckpointManager(tmp_path)
    resumed.load(tmp_path / "last.pt", model, optimizer, scheduler)
    assert resumed.best_f1 == .75
    assert resumed.best_step == 12


def test_legacy_save_cadence_is_removed_from_resolved_config():
    cfg = TrainConfig.from_dict({'checkpoint': {'save_every_steps': 42}})
    assert 'save_every_steps' not in cfg.to_dict()['checkpoint']


def test_a_checkpoint_carries_and_restores_python_numpy_and_device_rng(tmp_path):
    """A resumed run must continue the *same* random stream, on every device.

    Dropout draws on the device RNG during training, so restoring only the torch CPU state
    would use different dropout masks -- ``deterministic: true`` pins operators, not the random
    stream.
    """
    import random

    import numpy as np

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    manager = CheckpointManager(tmp_path)

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    manager.save(model, optimizer, scheduler, 1, {})

    state = torch.load(tmp_path / 'step-1.pt', weights_only=False)['rng']
    assert {'python', 'torch', 'numpy'} <= set(state)
    if torch.cuda.is_available():
        assert 'torch_cuda' in state

    expected = (random.random(), float(np.random.rand()), torch.rand(3))
    # Burn the streams, then resume: load() restores every captured generator.
    random.random(), np.random.rand(), torch.rand(5)
    manager.load(tmp_path / 'step-1.pt', model, optimizer, scheduler)
    got = (random.random(), float(np.random.rand()), torch.rand(3))
    assert got[0] == expected[0]
    assert got[1] == expected[1]
    assert torch.equal(got[2], expected[2])
