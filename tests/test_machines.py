"""One `machine` field picks the model and data roots (utils/machines.py).

The paths used to live in a gitignored `scripts/env.local.sh`, so a run's locations were
invisible in the repository and a `git pull` could not tell whether the local file was still
correct. They are a table now, and a run names its machine; this file pins the resolution
rules, because getting them wrong silently points a run at the wrong data tree:

1. an explicitly named machine supplies MODEL_ROOT / DATA_ROOT / WANDB_MODE outright;
2. a machine recognised only from the hostname only *fills gaps* -- an exported MODEL_ROOT
   must never be overwritten by a guess;
3. an unknown name is an error listing the known ones, never a silent fallback;
4. the resolved machine is recorded on the config (and in `to_dict()`), so a run says where it
   ran.

Run with `python tests/test_machines.py` or `pytest tests`.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import TrainConfig, expand_env, expand_env_values  # noqa: E402
from utils.machines import (  # noqa: E402
    MACHINES,
    canonical_machine,
    fill_missing,
    machine_environ,
    machine_names,
    resolve_machine,
)

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "configs" / "4090" / "qwen-1.7b" / "baseline" / "train_baseline.yaml"
MACHINE_VARIABLES = ("MACHINE", "MODEL_ROOT", "DATA_ROOT", "WANDB_MODE")


class _Hidden:
    """Hide the ambient machine variables (and the hostname) for one test body."""

    def __init__(self, hostname: str = "no-such-machine"):
        self.hostname = hostname
        self.saved: dict[str, str | None] = {}

    def __enter__(self):
        import socket

        self.saved = {name: os.environ.pop(name, None) for name in MACHINE_VARIABLES}
        self.real_hostname = socket.gethostname
        socket.gethostname = lambda: self.hostname
        return self

    def __exit__(self, *exc):
        import socket

        socket.gethostname = self.real_hostname
        for name, value in self.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        return False


def test_the_table_holds_both_machines_with_distinct_paths():
    assert set(MACHINES) == {"4090", "h200"}
    for name, entry in MACHINES.items():
        assert entry["MODEL_ROOT"].startswith("/"), name
        assert entry["DATA_ROOT"].startswith("/"), name
        assert entry["DATA_ROOT"].endswith("aggregated"), name
    assert MACHINES["4090"]["MODEL_ROOT"] != MACHINES["h200"]["MODEL_ROOT"]
    # The H200 has no internet, so its entry has to keep wandb offline.
    assert MACHINES["h200"]["WANDB_MODE"] == "offline"


def test_names_and_aliases_resolve_to_the_canonical_machine():
    assert canonical_machine("h200") == "h200"
    assert canonical_machine(" 4090 ") == "4090"
    assert canonical_machine("4x4090") == "4090"
    assert canonical_machine("xmu90") == "4090"
    assert canonical_machine(None) is None
    assert canonical_machine("") is None
    assert "h200" in machine_names() and "4x4090" in machine_names()


def test_an_unknown_machine_is_an_error_not_a_silent_fallback():
    for name in ("h100", "4090x", "laptop"):
        try:
            canonical_machine(name)
        except ValueError as error:
            assert name in str(error) and "4090" in str(error) and "h200" in str(error), error
        else:
            raise AssertionError(f"{name!r} must not resolve to a machine")


def test_an_explicit_machine_supplies_the_paths_even_over_an_exported_one():
    with _Hidden():
        os.environ["MODEL_ROOT"] = "/somewhere/else"
        cfg = TrainConfig.from_file(CONFIG, machine="h200")
        assert cfg.machine == "h200"
        assert cfg.data.root == MACHINES["h200"]["DATA_ROOT"]
        assert cfg.model.name_or_path.startswith(MACHINES["h200"]["MODEL_ROOT"] + "/models--")
        # ... and the config records it for the run's wandb entry / checkpoint.
        assert cfg.to_dict()["machine"] == "h200"


def test_the_machine_field_in_the_config_is_honoured(tmp_path):
    with _Hidden("no-such-machine"):
        text = CONFIG.read_text(encoding="utf-8").replace(
            "model:", "machine: h200\nmodel:", 1
        )
        path = tmp_path / "with_machine.yaml"
        path.write_text(text, encoding="utf-8")
        cfg = TrainConfig.from_file(path)
        assert cfg.machine == "h200"
        assert cfg.data.root == MACHINES["h200"]["DATA_ROOT"]


def test_the_environment_variable_names_the_machine_before_the_hostname():
    with _Hidden(hostname="xmu90"):
        os.environ["MACHINE"] = "h200"
        cfg = TrainConfig.from_file(CONFIG)
        assert cfg.machine == "h200"
        assert cfg.data.root == MACHINES["h200"]["DATA_ROOT"]


def test_a_hostname_only_fills_the_variables_the_environment_did_not_set():
    with _Hidden(hostname="gpu-lg-cmc-h-h200-0117"):
        cfg = TrainConfig.from_file(CONFIG)
        # Guessed, so it may fill both.
        assert cfg.machine == "h200"
        assert cfg.data.root == MACHINES["h200"]["DATA_ROOT"]

    with _Hidden(hostname="gpu-lg-cmc-h-h200-0117"):
        os.environ["MODEL_ROOT"] = "/my/own/weights"
        cfg = TrainConfig.from_file(CONFIG)
        # A guess must not replace a deliberate export ...
        assert cfg.model.name_or_path.startswith("/my/own/weights/models--")
        # ... while the variable nobody set still comes from the table.
        assert cfg.data.root == MACHINES["h200"]["DATA_ROOT"]


def test_an_unrecognised_hostname_changes_nothing():
    with _Hidden(hostname="some-other-box"):
        cfg = TrainConfig.from_file(CONFIG)
        assert cfg.machine is None
        assert cfg.data.root == "/data/lz/contexts/aggregated"
        assert cfg.model.name_or_path.startswith("/data/lz/hf_cache/hub/models--")


def test_expansion_overrides_beat_the_environment():
    with _Hidden():
        os.environ["MODEL_ROOT"] = "/from/environment"
        assert expand_env("${MODEL_ROOT:-/fallback}") == "/from/environment"
        assert expand_env("${MODEL_ROOT:-/fallback}", {"MODEL_ROOT": "/from/machine"}) == "/from/machine"
        assert expand_env_values({"a": "${DATA_ROOT:-/x}"}, {"DATA_ROOT": "/y"}) == {"a": "/y"}


def test_fill_missing_leaves_exported_variables_alone():
    values = machine_environ("h200")
    assert fill_missing(values, {}) == values
    assert "MODEL_ROOT" not in fill_missing(values, {"MODEL_ROOT": "/exported"})


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            argument = {}
            import inspect

            if inspect.signature(function).parameters:
                import tempfile

                argument = {"tmp_path": Path(tempfile.mkdtemp())}
            try:
                function(**argument)
            except Exception as error:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(error).__name__}: {error}")
            else:
                print(f"ok   {name}")
    print("\nVERDICT:", "ALL PASSED" if not failures else f"{failures} FAILED")
    raise SystemExit(1 if failures else 0)
