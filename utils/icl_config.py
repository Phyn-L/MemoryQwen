"""Validated YAML defaults for the existing ICL evaluator."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import yaml

from .model_paths import machine_paths
from .config import expand_env_values


@dataclass
class ICLModel:
    name: str = 'Qwen3-1.7B'
    dtype: str = 'bfloat16'


@dataclass
class ICLData:
    datasets: list[str] = field(default_factory=lambda: ['squad'])
    split: str = 'validation'
    demo_split: str = 'train'
    root: str = '${DATA_ROOT:-/data/lz/contexts/aggregated}'
    max_samples: int | None = None
    max_contexts: int | None = None
    max_context_tokens: int | None = None


@dataclass
class ICLInference:
    batch_size: int = 2
    num_shots: int = 4
    max_input_tokens: int = 8192
    max_new_tokens: int = 32
    seed: int = 42
    num_workers: int = 0
    use_chat_template: bool = True


def load_icl_defaults(path, machine=None):
    from utils.config_paths import resolve_config_path
    raw = yaml.safe_load(resolve_config_path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError('ICL config must be a mapping')
    unknown = set(raw) - {'model', 'data', 'inference', 'output_dir', 'machine'}
    if unknown:
        raise ValueError(f'Unknown ICL config sections: {sorted(unknown)}')
    resolved, paths = machine_paths(machine or raw.get('machine'))
    raw = expand_env_values(raw, paths)
    model = ICLModel(**raw.get('model', {}))
    data = ICLData(**raw.get('data', {}))
    inference = ICLInference(**raw.get('inference', {}))
    if not isinstance(data.datasets, list) or not data.datasets or any(not isinstance(x, str) or not x or '/' in x or x in ('.', '..') for x in data.datasets):
        raise ValueError('data.datasets must be a non-empty list of dataset directory names')
    if data.split not in ('train', 'validation', 'test') or data.demo_split != 'train':
        raise ValueError('Use split train/validation/test and demo_split=train')
    if model.dtype not in ('bfloat16', 'float16', 'float32'):
        raise ValueError('Unsupported model.dtype')
    for name in ('max_samples', 'max_contexts', 'max_context_tokens'):
        value = getattr(data, name)
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError(f'data.{name} must be a positive integer or null')
    return dict(model=model.name, dtype=model.dtype, datasets=data.datasets,
                split=data.split, data_root=data.root, machine=resolved,
                max_samples=data.max_samples, max_contexts=data.max_contexts,
                max_context_tokens=data.max_context_tokens,
                output_dir=raw.get('output_dir'), **{
                    ('no_chat_template' if k == 'use_chat_template' else k):
                    (not v if k == 'use_chat_template' else v)
                    for k, v in asdict(inference).items()})
