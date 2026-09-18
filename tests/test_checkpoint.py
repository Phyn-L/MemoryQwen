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


def test_legacy_save_cadence_is_removed_from_resolved_config():
    cfg = TrainConfig.from_dict({'checkpoint': {'save_every_steps': 42}})
    assert 'save_every_steps' not in cfg.to_dict()['checkpoint']
