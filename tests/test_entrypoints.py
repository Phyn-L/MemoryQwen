"""Offline checks for checkpoint-driven evaluation, local models and YAML entrypoints."""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.config import TrainConfig
from utils.icl_config import load_icl_defaults
from utils.launcher import command
from utils.machines import MACHINES
from utils.model_paths import relocate_saved_paths, resolve_model_path


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_preserves_architecture_and_rebases_machine(tmp_path):
    cfg = TrainConfig.from_file(ROOT / 'configs/ablations/memory_length.yaml', machine='4090')
    checkpoint = tmp_path / 'last.pt'
    torch.save({'config': cfg.to_dict()}, checkpoint)
    actual = script('test').evaluation_config(checkpoint, machine='h200')
    assert actual.memory == cfg.memory
    assert actual.model.name_or_path.endswith(cfg.model.name_or_path.split('/snapshots/')[1])
    assert actual.model.name_or_path.startswith(MACHINES['h200']['MODEL_ROOT'])
    assert actual.data.root == MACHINES['h200']['DATA_ROOT']
    preset = ROOT / 'configs/test.yaml'
    assert script('test').evaluation_config(checkpoint, preset, '4090').memory.memory_length == 32


@pytest.mark.parametrize('mode', ['memory_question', 'memory_only', 'learned_prefix', 'off'])
@pytest.mark.parametrize('preset', [None, ROOT / 'configs/test.yaml'])
def test_checkpoint_preserves_readout_mode(tmp_path, mode, preset):
    cfg = TrainConfig.from_file(ROOT / 'configs/train_baseline.yaml')
    cfg.memory.readout_mode = mode
    cfg.memory.readout_length = 8
    checkpoint = tmp_path / 'reader.pt'
    torch.save({'config': cfg.to_dict()}, checkpoint)
    actual = script('test').evaluation_config(checkpoint, config=preset)
    actual.validate()
    assert actual.memory == cfg.memory


def test_legacy_checkpoint_defaults_to_question_readout(tmp_path):
    cfg = TrainConfig.from_file(ROOT / 'configs/train_baseline.yaml').to_dict()
    cfg['memory'].pop('readout_mode')
    cfg['memory']['readout_length'] = 8
    checkpoint = tmp_path / 'legacy.pt'
    torch.save({'config': cfg}, checkpoint)
    actual = script('test').evaluation_config(checkpoint)
    actual.validate()
    assert actual.memory.readout_mode == 'memory_question'
    assert actual.memory.readout_length == 8


def test_checkpoint_without_config_fails(tmp_path):
    p = tmp_path / 'empty.pt'
    torch.save({'model': {}}, p)
    with pytest.raises(ValueError, match='no saved config'):
        script('test').evaluation_config(p)


def test_model_resolution_uses_ref_and_rejects_ambiguous_cache(tmp_path):
    cache = tmp_path / 'models--Qwen--Qwen3-8B'
    for name in ['first', 'second']:
        p = cache / 'snapshots' / name
        p.mkdir(parents=True)
        (p / 'config.json').write_text('{}')
    with pytest.raises(ValueError, match='specify its full path'):
        resolve_model_path('qwen3-8b', model_root=tmp_path)
    (cache / 'refs').mkdir()
    (cache / 'refs/main').write_text('first')
    assert resolve_model_path('Qwen/Qwen3-8B', model_root=tmp_path).endswith('/first')
    (cache / 'refs/main').write_text('missing')
    with pytest.raises(FileNotFoundError, match='incomplete'):
        resolve_model_path('Qwen3-8B', model_root=tmp_path)


def test_stale_checkpoint_model_path_resolves_by_name(tmp_path, monkeypatch):
    model = tmp_path / 'models--Qwen--Qwen3-1.7B'
    snapshot = model / 'snapshots' / 'revision-new'
    snapshot.mkdir(parents=True)
    (snapshot / 'config.json').write_text('{}')
    monkeypatch.setitem(MACHINES['h200'], 'MODEL_ROOT', str(tmp_path))
    raw = {
        'model': {'name_or_path': '/old/machine/models--Qwen--Qwen3-1.7B/snapshots/revision-old'},
        'data': {'root': MACHINES['h200']['DATA_ROOT']},
    }
    actual = relocate_saved_paths(raw, 'h200')
    assert actual['model']['name_or_path'] == str(snapshot)
    assert raw['model']['name_or_path'].endswith('revision-old')


def test_icl_yaml_cli_precedence_and_dataset_names(tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    module = script('evaluation/test_icl_suite')
    args = module.parse_args(['--config', str(ROOT / 'configs/icl_zeroshot.yaml'),
                              '--machine', 'h200', '--model', str(model), '--bs', '7',
                              '--datasets', 'hotpotqa(distractor)'])
    assert args.batch_size == 7 and args.num_shots == 0
    assert args.datasets == ['hotpotqa(distractor)'] and args.max_new_tokens == 32
    assert args.data_root == MACHINES['h200']['DATA_ROOT']
    # The file itself comes from src/eval_data.py; only the layout module may build it.
    assert module.slug('hotpotqa(distractor)') == 'hotpotqa_distractor'
    assert args.model == str(model)
    with pytest.raises(SystemExit):
        # Old aggregated names are gone, and a bad batch size is still rejected.
        module.parse_args(['--config', str(ROOT / 'configs/icl_zeroshot.yaml'),
                           '--datasets', 'hotpotqa'])
    with pytest.raises(SystemExit):
        module.parse_args(['--config', str(ROOT / 'configs/icl_zeroshot.yaml'), '--bs', '0'])


def test_launch_uses_current_interpreter_and_gpu_count(monkeypatch):
    monkeypatch.delenv('NUM_PROCESSES', raising=False)
    args = ['--config', 'path with spaces/train.yaml', '--machine', 'h200']
    cmd = command('train', args, 4)
    assert cmd[:3] == [sys.executable, '-m', 'torch.distributed.run']
    assert '--nproc-per-node=4' in cmd and cmd[-4:] == args
    assert '-m' not in command('test', ['--help'], 0)
    monkeypatch.setenv('NUM_PROCESSES', '8')
    with pytest.raises(ValueError, match='only 4'):
        command('train', [], 4)


def test_presets_are_valid():
    for p in [ROOT / 'configs/train_baseline.yaml', *ROOT.glob('configs/ablations/*.yaml')]:
        cfg = TrainConfig.from_file(p)
        cfg.validate()
    for p in ROOT.glob('configs/icl_zeroshot.yaml'):
        assert load_icl_defaults(p)['batch_size'] > 0


def test_unknown_icl_sections_fail(tmp_path):
    p = tmp_path / 'bad.yaml'
    p.write_text('infernce: {}\n')
    with pytest.raises(ValueError, match='Unknown'):
        load_icl_defaults(p)


def test_icl_machine_defaults_preserve_explicit_model(monkeypatch):
    module = script('evaluation/test_icl_suite')
    monkeypatch.setattr(module, 'resolve_model_path', lambda name, machine: name)
    args = module.parse_args(['--machine', 'h200'])
    assert args.model.startswith(MACHINES['h200']['MODEL_ROOT'] + '/')
    assert args.data_root == MACHINES['h200']['DATA_ROOT']
    explicit = module.DEFAULT_MODEL
    args = module.parse_args(['--machine', 'h200', '--model', explicit])
    assert args.model == explicit
