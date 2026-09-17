"""The training loop's schedule: configurable cadences, a final evaluation, a clean allocator.

Four behaviours a single-epoch run depends on, and that are easy to lose in a refactor:

1. the loss-logging cadence is ``logging.log_every``, not a hard-coded 10;
2. an evaluation whose cadence does not divide the run still happens on the last step, because
   on a one-epoch run that final number is the whole result;
3. the trainer echoes the resolved schedule -- steps, global batch, every cadence -- before it
   spends compute, so a batch-size or process-count mismatch is visible in one line;
4. the allocator cache is handed back after an evaluation. The H200 ON arm died in the *next*
   training step's backward with "5.31 GiB reserved but unallocated" while asking for 3.40 GiB,
   immediately after its first evaluation at step 500.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils.config import LoggingConfig, TrainConfig  # noqa: E402

TRAIN_PY = REPO / "scripts" / "train.py"
CONFIG = REPO / "configs" / "qwen-1.7b" / "ab_h200_on.yaml"
STEPS_PER_EPOCH = 7890


def _train_source() -> str:
    return TRAIN_PY.read_text(encoding="utf-8")


def _train_module():
    spec = importlib.util.spec_from_file_location("_schedule_train_script", TRAIN_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_schedule_train_script"] = module
    spec.loader.exec_module(module)
    return module


def test_log_every_defaults_to_the_historical_resolution_and_is_validated():
    assert LoggingConfig().log_every == 10

    broken = CONFIG.read_text(encoding="utf-8").replace("log_every: 25", "log_every: 0")
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "broken.yaml"
        path.write_text(broken, encoding="utf-8")
        try:
            TrainConfig.from_file(path).validate()
        except ValueError as error:
            assert "log_every" in str(error)
        else:
            raise AssertionError("logging.log_every = 0 must be rejected")


def test_the_last_step_is_always_evaluated():
    """500 over 7,890 fires last at 7,500 (95%), 1000 at 7,000 (89%): the end must be scored."""
    train = _train_module()
    assert train.should_evaluate(500, 500, STEPS_PER_EPOCH)
    assert not train.should_evaluate(501, 500, STEPS_PER_EPOCH)
    assert train.should_evaluate(STEPS_PER_EPOCH, 500, STEPS_PER_EPOCH)
    assert train.should_evaluate(STEPS_PER_EPOCH, 1000, STEPS_PER_EPOCH)
    # ... and not one step earlier, or every step would end up evaluating.
    assert not train.should_evaluate(STEPS_PER_EPOCH - 1, 1000, STEPS_PER_EPOCH)


def test_the_loop_takes_its_cadences_from_the_config():
    source = _train_source()
    assert "step % cfg.logging.log_every == 0" in source
    assert "step % 10 == 0" not in source, "the loss-logging cadence must come from the config"
    assert source.count("should_evaluate(") >= 3, "both evaluations must use the final-step rule"


def test_the_loop_echoes_the_resolved_schedule():
    source = _train_source()
    assert '"schedule: "' in source
    for field in ("steps=", "ranks=", "warmup=", "save_every_steps=", "log_every="):
        assert field in source, field


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
