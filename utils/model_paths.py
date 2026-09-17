"""Resolve local Qwen checkpoints without downloading or guessing among revisions."""
from pathlib import Path
import os
import re

from .machines import fill_missing, machine_environ, resolve_machine


def machine_paths(machine=None):
    resolved, authoritative = resolve_machine(machine)
    values = machine_environ(resolved)
    if not authoritative:
        values = fill_missing(values, os.environ)
    return resolved, {
        'MODEL_ROOT': values.get('MODEL_ROOT', os.environ.get('MODEL_ROOT', '/data/lz/hf_cache/hub')),
        'DATA_ROOT': values.get('DATA_ROOT', os.environ.get('DATA_ROOT', '/data/lz/contexts/aggregated')),
    }


def resolve_model_path(name, machine=None, model_root=None):
    """Accept an explicit directory, Qwen/name, or a case-insensitive Qwen family name.

    A cache's refs/main wins; without it exactly one complete snapshot is required.
    Ambiguous revisions fail instead of silently selecting different weights.
    """
    path = Path(name).expanduser()
    if path.is_dir() and (path / 'config.json').is_file():
        return str(path.resolve())
    if path.is_absolute() or name.startswith(('.', '~')):
        raise FileNotFoundError(f'Model directory has no config.json: {path}')
    model_name = name.removeprefix('Qwen/').removeprefix('qwen/')
    if not re.fullmatch(r'Qwen[\w.\-]+', model_name, re.IGNORECASE):
        raise ValueError(f'Use a Qwen model name (e.g. Qwen3-8B) or a local directory: {name}')
    root = Path(model_root or machine_paths(machine)[1]['MODEL_ROOT'])
    names = {model_name.lower(), f'models--qwen--{model_name}'.lower()}
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.lower() in names] if root.is_dir() else []
    if len(candidates) != 1:
        raise FileNotFoundError(f'Expected one local directory for {name} under {root}; found {candidates}')
    path = candidates[0]
    if (path / 'config.json').is_file():
        return str(path.resolve())
    ref = path / 'refs/main'
    if ref.is_file():
        snapshot = path / 'snapshots' / ref.read_text().strip()
        if not (snapshot / 'config.json').is_file():
            raise FileNotFoundError(f'refs/main points to an incomplete snapshot: {snapshot}')
        return str(snapshot.resolve())
    snapshots = sorted(p.parent for p in (path / 'snapshots').glob('*/config.json'))
    if len(snapshots) != 1:
        raise ValueError(f'{name}: expected one snapshot, found {len(snapshots)}; specify its full path')
    return str(snapshots[0].resolve())


def relocate_saved_paths(raw, machine):
    """Rebase known machine roots, preserving the saved model revision and data suffix."""
    from copy import deepcopy
    from .machines import MACHINES
    values = deepcopy(raw)
    resolved, paths = machine_paths(machine)
    for section, field, root_key in [('model', 'name_or_path', 'MODEL_ROOT'), ('data', 'root', 'DATA_ROOT')]:
        old = values.get(section, {}).get(field)
        if not old or '${' in old:
            continue
        for entry in MACHINES.values():
            try:
                suffix = Path(old).relative_to(entry[root_key])
            except ValueError:
                continue
            values[section][field] = str(Path(paths[root_key]) / suffix)
            break
        else:
            # An unknown absolute path must not accidentally select another backbone/data tree.
            if Path(old).is_absolute() and not Path(old).exists():
                raise ValueError(f'Cannot relocate {section}.{field}={old!r}; use --config with explicit paths')
    values['machine'] = resolved
    return values
