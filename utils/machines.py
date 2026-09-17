"""Per-machine locations, so a run needs one ``machine`` field instead of an env file.

The Qwen weights and the aggregated data tree sit in different places on the two machines
this project trains on, and that used to be a gitignored ``scripts/env.local.sh`` exporting
``MODEL_ROOT`` / ``DATA_ROOT``. Keeping the table in the repository makes a run
self-describing: ``machine: h200`` in the config, ``--machine h200`` on the command line, or
``MACHINE=h200`` in the environment is enough, and nothing machine-specific has to be written
into a tracked YAML -- which is what turned every ``git pull`` into a conflict.

The keys are the environment variables the configs already interpolate
(``${MODEL_ROOT:-...}``), so this table is a *default provider* for them rather than a second
mechanism:

* a machine named explicitly (CLI / ``machine:`` / ``MACHINE``) supplies its values outright;
* a machine recognised from the hostname only fills variables that are not already set, so it
  can never overwrite a deliberate ``export MODEL_ROOT=...``;
* with neither, the configs keep using their own defaults (and ``scripts/env.local.sh``, where
  it still exists, keeps working as before).

Edit ``MACHINES`` to add a machine.
"""
from __future__ import annotations

import os
import socket
from typing import Mapping

MACHINES: dict[str, dict[str, str]] = {
    # 4x RTX 4090: the tracked configs' own ${MODEL_ROOT:-...} defaults apply here, and the
    # entry makes the machine name explicit.
    "4090": {
        "MODEL_ROOT": "/data/lz/hf_cache/hub",
        "DATA_ROOT": "/data/lz/contexts/aggregated",
    },
    # 8x H200, no internet, so W&B has to stay offline.
    "h200": {
        "MODEL_ROOT": "/home/lijie/proj2/xmu/lz",
        "DATA_ROOT": "/home/lijie/proj2/xmu/lz/aggregated",
        "WANDB_MODE": "offline",
    },
}

# Names that appear in hostnames or that people type for the same machine.
ALIASES: dict[str, str] = {
    "4x4090": "4090",
    "xmu90": "4090",          # the 4090 box's hostname
    "h200-gateway": "h200",
}


def machine_names() -> tuple[str, ...]:
    """Every accepted spelling, canonical entries first."""
    return tuple(MACHINES) + tuple(sorted(ALIASES))


def canonical_machine(name: str | None) -> str | None:
    """Normalise one machine name; ``None`` when nothing was given, error when unknown."""
    if name is None or not str(name).strip():
        return None
    key = str(name).strip()
    key = ALIASES.get(key, key)
    if key not in MACHINES:
        raise ValueError(
            f"unknown machine {name!r}; known machines are {', '.join(machine_names())} "
            "(add an entry to utils/machines.py::MACHINES)"
        )
    return key


def machine_environ(name: str | None) -> dict[str, str]:
    """The environment values one machine provides (``{}`` when no machine was named)."""
    canonical = canonical_machine(name)
    if canonical is None:
        return {}
    return dict(MACHINES[canonical])


def resolve_machine(explicit: str | None = None) -> tuple[str | None, bool]:
    """``(machine, authoritative)`` from the CLI/config, ``MACHINE``, or the hostname.

    ``authoritative`` is True when the machine was *named* -- by the caller, by a ``machine:``
    field or by the ``MACHINE`` variable -- and False when it was only recognised from the
    hostname. Callers use that to decide whether the table may replace an exported
    ``MODEL_ROOT``/``DATA_ROOT``: a named machine may, a guessed one may not.
    """
    canonical = canonical_machine(explicit)
    if canonical is not None:
        return canonical, True
    from_environment = canonical_machine(os.environ.get("MACHINE"))
    if from_environment is not None:
        return from_environment, True
    hostname = _hostname().lower()
    if not hostname:
        return None, False
    for candidate in machine_names():
        canonical_candidate = canonical_machine(candidate)
        if canonical_candidate and candidate.lower() in hostname:
            return canonical_candidate, False
    return None, False


def fill_missing(machine_values: Mapping[str, str], environ: Mapping[str, str]) -> dict[str, str]:
    """Drop the values whose variable is already set in ``environ``."""
    return {key: value for key, value in machine_values.items() if not environ.get(key)}


def _hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:  # pragma: no cover - a container without a hostname
        return ""
