"""The reader A/B pair must differ only in the six reader switches.

An A/B is only readable if the two arms are identical everywhere else. These configs are
two files, so a later edit to one of them (a cadence, a context length, a loss weight)
silently turns the comparison into "whatever changed" -- which is exactly how the previous
pair of runs (0rj6x1xc at ctx=2048/M=64 and vry7n1sw at ctx=512/M=16) ended up
uninterpretable. So the diff is asserted here rather than left to review.

The schedule is pinned too: both arms are one epoch of 3,945 steps (all @ ctx=1024 with a
global batch of 8 x 8 ranks = 64), and every cadence is written against that budget. The loop
that consumes those numbers is tested in tests/test_train_schedule.py.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils.config import TrainConfig  # noqa: E402

ON_CONFIG = REPO / "configs" / "qwen-1.7b" / "ab_h200_on.yaml"
OFF_CONFIG = REPO / "configs" / "qwen-1.7b" / "ab_h200_off.yaml"

# The switches the ON arm turns on, with the value the OFF arm must carry. Everything else
# -- including checkpoint.output_dir, which only exists so the two runs do not overwrite
# each other -- has to match.
SWITCHES = {
    "head_mode": ("tied", "linear"),
    "init_mode": ("token_embed", "randn"),
    "allow_slot_attention": (True, False),
    "ae_lm_weight": (1.0, 0.0),
    "distill_weight": (0.3, 0.0),
    "readout_length": (8, 0),
}

# 1 epoch over the 252,465 contexts kept by train_datasets=all at ctx=1024, global batch
# 8 x 8 ranks = 64 -> ceil(252465 / 64) = 3,945 steps. Second round: M raised to 64 (16:1,
# the compression ratio the M=64/ctx=2048 arm won with) and max_answer_tokens cut to 64 to
# buy back the QA branch's peak (its full-vocab logits halve).
STEPS_PER_EPOCH = 3945
SCHEDULE = {
    "training.batch_size": 8,
    "training.epochs": 1,
    "scheduler.warmup_steps": 200,
    "evaluation.teacher_forced_every": 200,
    "evaluation.autoregressive_every": 400,
    "checkpoint.save_every_steps": 400,
    "logging.log_every": 25,
    "data.max_context_tokens": 1024,
    "data.max_answer_tokens": 64,
    "memory.memory_length": 64,
    "data.validation_max_samples": 2000,
    "evaluation.autoregressive_max_qa": 1024,
    "evaluation.max_new_tokens": 32,
}


def _load(path: Path) -> TrainConfig:
    config = TrainConfig.from_file(path)
    config.validate()
    return config


def _flatten(config: TrainConfig) -> dict:
    flat = {}
    for section, values in asdict(config).items():
        if not isinstance(values, dict):
            # Plain top-level fields (`machine`) sit next to the config sections.
            flat[section] = values
            continue
        for key, value in values.items():
            flat[f"{section}.{key}"] = value
    return flat


def test_both_arms_are_valid_and_scheduled_for_one_epoch():
    for path in (ON_CONFIG, OFF_CONFIG):
        config = _load(path)
        assert config.training.epochs == 1, path


def test_the_arms_differ_only_in_the_six_switches_and_the_output_dir():
    on = _flatten(_load(ON_CONFIG))
    off = _flatten(_load(OFF_CONFIG))

    assert set(on) == set(off)
    differing = {key for key in on if on[key] != off[key]}
    assert differing == {"memory." + key for key in SWITCHES} | {"checkpoint.output_dir"}, differing


def test_the_switches_have_the_documented_values():
    on, off = _load(ON_CONFIG), _load(OFF_CONFIG)
    for key, (on_value, off_value) in SWITCHES.items():
        assert getattr(on.memory, key) == on_value, key
        assert getattr(off.memory, key) == off_value, key


def test_every_cadence_is_written_against_the_single_epoch_budget():
    for path in (ON_CONFIG, OFF_CONFIG):
        flat = _flatten(_load(path))
        for key, expected in SCHEDULE.items():
            assert flat[key] == expected, (path.name, key, flat[key], expected)
    # Nothing may ask for more steps than the epoch has, and each cadence must fire a
    # useful number of times inside it.
    assert STEPS_PER_EPOCH // SCHEDULE["evaluation.teacher_forced_every"] >= 10
    assert STEPS_PER_EPOCH // SCHEDULE["evaluation.autoregressive_every"] >= 5
    assert STEPS_PER_EPOCH // SCHEDULE["checkpoint.save_every_steps"] >= 3
    assert SCHEDULE["scheduler.warmup_steps"] < STEPS_PER_EPOCH // 10


def test_the_expected_step_count_matches_the_launcher():
    """scripts/run_ab.sh prints the same budget the schedule was written for."""
    source = (REPO / "scripts" / "run_ab.sh").read_text(encoding="utf-8")
    assert "CONTEXTS=252465" in source
    assert 252465 // 64 + (252465 % 64 > 0) == STEPS_PER_EPOCH


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
