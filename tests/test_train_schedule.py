"""The training loop's schedule: configurable cadences, a final evaluation, a clean allocator.

Four behaviours a single-epoch run depends on, and that are easy to lose in a refactor:

1. the loss-logging cadence is ``logging.log_every``, not a hard-coded 10;
2. an evaluation whose cadence does not divide the run still happens on the last step, because
   on a one-epoch run that final number is the whole result;
3. the trainer echoes the resolved schedule -- steps, global batch, every cadence -- before it
   spends compute, so a batch-size or process-count mismatch is visible in one line;
4. the allocator cache is handed back after an evaluation. The H200 ON arm died in the *next*
   training step's backward with "5.31 GiB reserved but unallocated" while asking for 3.40 GiB,
   immediately after its first evaluation at step 500;
5. the LR scheduler is *not* handed to ``accelerator.prepare`` (that wraps it and advances it
   once per rank, which compressed every multi-GPU schedule by the rank count), and the LR is
   logged so a distorted schedule is visible in the run history.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils.config import LoggingConfig, TrainConfig  # noqa: E402

TRAIN_PY = REPO / "scripts" / "train.py"
CONFIG = REPO / "configs/train_baseline.yaml"
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

    # Built from a dict on purpose: patching the tracked YAML's text made this test depend on
    # a literal ("log_every: 25") that a config edit silently moved out from under it.
    cfg = TrainConfig.from_dict({"logging": {"log_every": 0}})
    with pytest.raises(ValueError, match="log_every"):
        cfg.validate()


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
    assert source.count("should_evaluate(") >= 2, "autoregressive evaluation must use the final-step rule"
    assert "teacher_metrics = evaluator.teacher_forced" not in source
    assert "include_teacher_metrics" not in source


def test_the_loop_echoes_the_resolved_schedule():
    source = _train_source()
    assert '"schedule: "' in source
    for field in ("steps=", "ranks=", "warmup=", "log_every="):
        assert field in source, field


def test_the_allocator_cache_is_released_after_an_evaluation():
    source = _train_source()
    assert "torch.cuda.empty_cache()" in source, (
        "the evaluation's cached blocks must not be left for the next backward"
    )
    # It must be tied to an evaluation having happened, not run every step.
    assert "metrics is not None" in source


def _parenthesised_call(source: str, needle: str) -> str:
    """The full text of the call starting at ``needle`` (balanced parentheses)."""
    start = source.index(needle)
    depth = 0
    for index in range(start, len(source)):
        if source[index] == "(":
            depth += 1
        elif source[index] == ")":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError(f"unbalanced call for {needle!r}")


def test_the_scheduler_is_not_wrapped_by_accelerate():
    """The LR schedule must advance once per optimizer step, not once per rank.

    ``accelerator.prepare`` replaces an LRScheduler with ``AcceleratedScheduler``, whose
    ``step()`` calls the wrapped scheduler ``num_processes`` times when ``split_batches``
    is False -- the default. With 8 ranks that consumes a cosine built for the whole run
    in the first eighth of it, and HF's cosine keeps being evaluated past its end, so the
    LR then oscillates between zero and the peak instead of annealing. Measured on 2 ranks
    with a cosine built for 8 steps: ``last_epoch`` went 2, 4, 6, 8 and the LR was 0 after
    four loop steps. The scheduler therefore stays unwrapped and the loop steps it.
    """
    source = _train_source()
    prepared = _parenthesised_call(source, "accelerator.prepare(")
    assert "scheduler" not in prepared, (
        "passing the scheduler to accelerator.prepare() compresses the LR schedule by the "
        f"rank count; prepared call was: {prepared}"
    )
    assert source.count("scheduler.step()") == 1, "the loop owns the single scheduler step"
    assert '"train/lr"' in source, "the LR must be logged, or a distorted schedule is invisible"


def test_training_metrics_include_ppl_and_grad_norm_without_mfu():
    source = _train_source()
    assert '"loss/ppl"' in source
    assert '"loss/qa_ppl"' in source
    assert '"loss/recon_ppl"' in source
    assert '"grad_norm/total"' in source
    assert '"train/MFU"' not in source
    assert "_gpu_peak_flops" not in source


def test_training_payload_groups_data_probabilities_and_drops_row_throughput_metrics():
    source = _train_source()
    assert '"data/qa_probability"' in source
    assert '"data/recon_probability"' in source
    for removed in (
        '"train/qa_rows"',
        '"train/recon_rows"',
        '"train/step_seconds"',
        '"train/supervised_tokens_per_second"',
        '"train/model_tokens_per_second"',
    ):
        assert removed not in source


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


def test_the_progress_line_carries_loss_qa_recon_and_ppl_at_once():
    """One postfix call, four fields: tqdm replaces the fields on every call.

    The loop used to set `{loss, ppl}` per micro-batch and then `{loss}` at every optimizer
    step, so with the default accumulation of 1 the perplexity was wiped on every step.
    """
    module = _train_module()
    # Sums over a two-micro-batch group (what `group_terms` holds at the boundary).
    terms = {"loss": 24.0, "qa_loss": 20.0, "recon_loss": 4.0, "distill_loss": 0.0}
    postfix = module.step_postfix(terms, 2)
    assert postfix["loss"] == "12.0000"
    assert postfix["qa"] == "10.0000"
    assert postfix["recon"] == "2.0000"
    assert postfix["ppl"] == "22026.47"
    assert list(postfix) == ["loss", "qa", "recon", "ppl"]

    # A single-micro-batch view is the running mean of the group, and the perplexity of a
    # large QA loss is clamped instead of overflowing.
    assert module.step_postfix(terms, 1)["loss"] == "24.0000"
    assert module.step_postfix({"loss": 1.0, "qa_loss": 1e6}, 1)["ppl"] == f"{math.exp(20.0):.2f}"


def test_the_loop_does_not_overwrite_the_postfix_with_loss_only():
    source = _train_source()
    assert "progress.set_postfix(step_postfix(group_terms, group_micro_batches))" in source
    assert "set_postfix(loss=" not in source, "a loss-only call wipes the other fields"
    assert "set_postfix(**" not in source, "keyword arguments would be sorted alphabetically"
