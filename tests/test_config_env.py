"""Machine-specific paths must be overridable without editing tracked files.

The cloud configs hardcoded `/data/lz/...` and the H200 has its weights and data under
`/home/lijie/proj2/xmu/lz`, so syncing meant editing tracked YAML every time -- and `git pull`
then conflicted on exactly those files. `${VAR:-default}` (see
`utils.config.expand_env`) removes the collision: the default still works where it was
written, and another machine overrides it through the environment or a gitignored
`scripts/env.local.sh`.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils.config import TrainConfig, expand_env, expand_env_values  # noqa: E402

CONFIGS = sorted((REPO / "configs").glob("*/train.yaml"))


def test_default_is_used_when_the_variable_is_unset():
    os.environ.pop("MEMORYQWEN_TEST_ROOT", None)
    assert expand_env("${MEMORYQWEN_TEST_ROOT:-/fallback}/x") == "/fallback/x"


def test_the_environment_wins_when_set():
    os.environ["MEMORYQWEN_TEST_ROOT"] = "/override"
    try:
        assert expand_env("${MEMORYQWEN_TEST_ROOT:-/fallback}/x") == "/override/x"
    finally:
        os.environ.pop("MEMORYQWEN_TEST_ROOT")


def test_an_empty_variable_falls_back_rather_than_producing_an_empty_path():
    os.environ["MEMORYQWEN_TEST_ROOT"] = ""
    try:
        assert expand_env("${MEMORYQWEN_TEST_ROOT:-/fallback}/x") == "/fallback/x"
    finally:
        os.environ.pop("MEMORYQWEN_TEST_ROOT")


def test_a_bare_reference_without_default_raises():
    os.environ.pop("MEMORYQWEN_TEST_ROOT", None)
    try:
        expand_env("${MEMORYQWEN_TEST_ROOT}/x")
    except ValueError as error:
        assert "MEMORYQWEN_TEST_ROOT" in str(error)
    else:
        raise AssertionError("an unset variable with no default must raise, not expand to empty")


def test_expansion_walks_nested_structures():
    os.environ["MEMORYQWEN_TEST_ROOT"] = "/override"
    try:
        assert expand_env_values({"a": ["${MEMORYQWEN_TEST_ROOT}/1", 2], "b": {"c": "plain"}}) == {
            "a": ["/override/1", 2], "b": {"c": "plain"},
        }
    finally:
        os.environ.pop("MEMORYQWEN_TEST_ROOT")


MACHINE_VARIABLES = ("MODEL_ROOT", "DATA_ROOT", "WANDB_MODE")


def _without_machine_variables():
    """Context manager: hide this machine's overrides so the defaults can be asserted.

    Without this the test only passes where the variables happen to be unset -- on the H200,
    where scripts/env.local.sh exports them, the shipped defaults are correctly overridden and
    the assertion failed. A test must not depend on the ambient environment.
    """
    saved = {name: os.environ.pop(name, None) for name in MACHINE_VARIABLES}

    class _Restore:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            return False

    return _Restore()


def test_cloud_defaults_resolve_without_any_environment():
    with _without_machine_variables():
        for config in CONFIGS:
            cfg = TrainConfig.from_file(config)
            cfg.validate()
            assert cfg.model.name_or_path.startswith("/data/lz/hf_cache/hub/models--Qwen--"), config
            assert cfg.data.root == "/data/lz/contexts/aggregated", config
            assert cfg.logging.wandb_mode == "online", config


def test_the_same_config_files_resolve_to_another_machine():
    """The exact H200 override, applied to every shipped config."""
    os.environ["MODEL_ROOT"] = "/home/lijie/proj2/xmu/lz"
    os.environ["DATA_ROOT"] = "/home/lijie/proj2/xmu/lz/aggregated"
    os.environ["WANDB_MODE"] = "offline"
    try:
        for config in CONFIGS:
            cfg = TrainConfig.from_file(config)
            cfg.validate()
            assert cfg.model.name_or_path.startswith("/home/lijie/proj2/xmu/lz/models--Qwen--"), config
            assert "/snapshots/" in cfg.model.name_or_path, config
            assert cfg.data.root == "/home/lijie/proj2/xmu/lz/aggregated", config
            assert cfg.logging.wandb_mode == "offline", config
    finally:
        for name in MACHINE_VARIABLES:
            os.environ.pop(name)


def test_the_icl_baseline_defaults_follow_the_same_variables():
    """The baseline script's own defaults must not be a second hardcoded copy."""
    source = (REPO / "scripts" / "test_icl_baseline.py").read_text(encoding="utf-8")
    assert "${MODEL_ROOT:-" in source and "${DATA_ROOT:-" in source


def test_env_local_is_gitignored_so_machine_settings_cannot_be_committed():
    ignore = (REPO / ".gitignore").read_text(encoding="utf-8")
    assert "scripts/env.local.sh" in ignore


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
