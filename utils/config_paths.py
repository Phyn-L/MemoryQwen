"""Resolve historical YAML names without retaining duplicate directory trees."""
import json
from pathlib import Path


def resolve_config_path(path):
    path = Path(path)
    if path.exists():
        return path
    root = Path(__file__).resolve().parent.parent
    key = str(path.relative_to(root)) if path.is_absolute() and path.is_relative_to(root) else path.as_posix()
    aliases = json.loads(Path(__file__).with_suffix('.json').read_text())
    return root / aliases[key] if key in aliases else path
