"""Offline checks for checkpoint-driven evaluation, local models and YAML entrypoints."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.config import TrainConfig
from utils.model_paths import resolve_model_path, relocate_saved_paths
from utils.machines import MACHINES
from utils.launcher import command
from utils.icl_config import load_icl_defaults


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_preserves_architecture_and_rebases_machine(tmp_path):
    cfg = TrainConfig.from_file(ROOT / 'configs/4090/qwen-1.7b/memory_length/train_reader-on_ctx1024_m32.yaml', machine='4090')
    checkpoint = tmp_path / 'last.pt'
    torch.save({'config': cfg.to_dict()}, checkpoint)
    actual = script('test').evaluation_config(checkpoint, machine='h200')
    assert actual.memory == cfg.memory
    assert actual.model.name_or_path.endswith(cfg.model.name_or_path.split('/snapshots/')[1])
    assert actual.model.name_or_path.startswith(MACHINES['h200']['MODEL_ROOT'])
    assert actual.data.root == MACHINES['h200']['DATA_ROOT']
    preset = ROOT / 'configs/4090/evaluation/test_hotpotqa.yaml'
    assert script('test').evaluation_config(checkpoint, preset, '4090').memory.memory_length == 32


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


def test_icl_yaml_cli_precedence_and_hotpot_paths(tmp_path):
    model = tmp_path / 'model'; model.mkdir(); (model / 'config.json').write_text('{}')
    module = script('test_icl_baseline')
    args = module.parse_args(['--config', str(ROOT / 'configs/4090/icl/icl_hotpotqa_4shot.yaml'),
                              '--machine', 'h200', '--model', str(model), '--bs', '7'])
    assert args.batch_size == 7 and args.num_shots == 4
    assert args.datasets == ['hotpotqa'] and args.max_new_tokens == 32
    assert module.dataset_paths(args)['hotpotqa'][0] == MACHINES['h200']['DATA_ROOT'] + '/hotpotqa/validation.jsonl'
    assert args.model == str(model)
    with pytest.raises(SystemExit):
        module.parse_args(['--config', str(ROOT / 'configs/4090/icl/icl_hotpotqa_4shot.yaml'), '--bs', '0'])


def test_hotpot_aggregated_records_shared_by_memory_and_icl(tmp_path):
    import json
    from src.icl_baseline import load_jsonl, format_prompt
    from src.data import AggregatedContextDataset
    folder = tmp_path / 'hotpotqa'; folder.mkdir()
    (folder / 'validation.jsonl').write_text(json.dumps({'context': 'Paris is in France.', 'qa_pairs': [
        {'id': 'hp1', 'question': 'Where is Paris?', 'answers': ['France'], 'metadata': {'supporting_facts': {'title': ['Paris'], 'sent_id': [0]}}} ]})+'\n')
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return type('Encoding', (), {'input_ids': list(range(len(text.split())))})()
    ds = AggregatedContextDataset(tmp_path, ['hotpotqa'], 'validation', Tokenizer(), 2048, None, True, True)
    icl = load_jsonl(folder / 'validation.jsonl', 'hotpotqa')
    assert len(ds) == len(icl) == 1
    assert icl[0].references == ('France',)
    assert 'Where is Paris?' in format_prompt(icl[0], [])


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


def test_presets_valid_and_legacy_aliases_equivalent():
    for p in ROOT.glob('configs/[4h]*/qwen-*/*/*.yaml'):
        cfg = TrainConfig.from_file(p)
        cfg.validate()
        if p.is_symlink():
            assert cfg.to_dict() == TrainConfig.from_file(p.resolve()).to_dict()
    for p in ROOT.glob('configs/4090/icl/*.yaml'):
        assert load_icl_defaults(p)['batch_size'] > 0


def test_unknown_icl_sections_fail(tmp_path):
    p = tmp_path / 'bad.yaml'; p.write_text('infernce: {}\n')
    with pytest.raises(ValueError, match='Unknown'):
        load_icl_defaults(p)


def test_icl_machine_defaults_preserve_explicit_model(monkeypatch):
    module = script('test_icl_baseline')
    monkeypatch.setattr(module, 'resolve_model_path', lambda name, machine: name)
    args = module.parse_args(['--machine', 'h200'])
    assert args.model.startswith(MACHINES['h200']['MODEL_ROOT'] + '/')
    assert args.squad_validation_file.startswith(MACHINES['h200']['DATA_ROOT'] + '/')
    explicit = module.DEFAULT_MODEL
    args = module.parse_args(['--machine', 'h200', '--model', explicit])
    assert args.model == explicit


def test_historical_training_paths_resolve():
    import json
    aliases = json.loads((ROOT / 'utils/config_paths.json').read_text())
    assert aliases
    for old, new in aliases.items():
        assert (ROOT / new).is_file()
        if '/evaluation/' in old:
            from utils.config_paths import resolve_config_path
            assert resolve_config_path(ROOT / old).read_bytes() == (ROOT / new).read_bytes()
        elif '/icl/' in old:
            assert load_icl_defaults(ROOT / old) == load_icl_defaults(ROOT / new)
        else:
            assert TrainConfig.from_file(ROOT / old).to_dict() == TrainConfig.from_file(ROOT / new).to_dict()
