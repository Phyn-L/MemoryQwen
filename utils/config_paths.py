"""Resolve an explicit configuration path; removed presets are not remapped."""
from pathlib import Path


def resolve_config_path(path):
    return Path(path)
