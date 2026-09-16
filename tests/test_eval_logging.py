"""W&B layout contract for the two evaluation modes.

W&B groups panels by the FIRST path component of a logged key, so the section layout is
decided entirely by `scripts/train.py:eval_log_payloads`. It used to log
`val/teacher_forced/*` and `val/autoregressive/*` -- one crowded `val` section -- plus a
`val/primary/*` mirror of the autoregressive numbers whose name made the headline result
look like it was called "primary". This file pins the intended layout:

* exactly two sections, one per evaluation mode;
* the same answer-quality keys in both, so they can be read side by side;
* loss-like scalars only under the teacher-forced section, even when they were computed by
  the autoregressive pass's internal teacher-forced run.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("_train_script", REPO / "scripts" / "train.py")
train = importlib.util.module_from_spec(spec)
sys.modules["_train_script"] = train
spec.loader.exec_module(train)

ANSWER_KEYS = set(train.ANSWER_METRIC_KEYS)
LOSS_KEYS = set(train.LOSS_KEYS)


def _sections(payloads):
    sections = {}
    for payload in payloads:
        for key, value in payload.items():
            section, _, metric = key.partition("/")
            assert _ and metric, f"key {key!r} has no section/metric split"
            assert "/" not in metric, f"key {key!r} is more than two levels deep"
            sections.setdefault(section, {})[metric] = value
    return sections


def _teacher_payload():
    values = {key: 0.5 for key in ANSWER_KEYS}
    values.update({key: 1.5 for key in LOSS_KEYS})
    return values


def _autoregressive_payload():
    values = {key: 0.7 for key in ANSWER_KEYS}
    values.update({key: 1.9 for key in LOSS_KEYS})  # copied from its internal TF pass
    return values


def test_both_modes_land_in_two_named_sections():
    payloads = train.eval_log_payloads(_teacher_payload(), _autoregressive_payload())
    sections = _sections(payloads)
    assert set(sections) == {train.TEACHER_FORCED_SECTION, train.AUTOREGRESSIVE_SECTION}
    assert train.TEACHER_FORCED_SECTION != train.AUTOREGRESSIVE_SECTION


def test_no_primary_section_remains():
    payloads = train.eval_log_payloads(_teacher_payload(), _autoregressive_payload())
    for key in (key for payload in payloads for key in payload):
        assert "primary" not in key, f"{key!r} still uses the old 'primary' naming"


def test_both_sections_report_the_same_answer_quality_keys():
    payloads = train.eval_log_payloads(_teacher_payload(), _autoregressive_payload())
    sections = _sections(payloads)
    teacher_keys = set(sections[train.TEACHER_FORCED_SECTION]) & ANSWER_KEYS
    autoregressive_keys = set(sections[train.AUTOREGRESSIVE_SECTION])
    assert teacher_keys == ANSWER_KEYS
    assert autoregressive_keys == ANSWER_KEYS, "the two sections must line up panel for panel"


def test_loss_scalars_stay_in_the_teacher_forced_section():
    # Both passes ran separately: loss keys appear once, under the teacher-forced section.
    payloads = train.eval_log_payloads(_teacher_payload(), _autoregressive_payload())
    sections = _sections(payloads)
    assert set(sections[train.AUTOREGRESSIVE_SECTION]) & LOSS_KEYS == set()
    assert LOSS_KEYS <= set(sections[train.TEACHER_FORCED_SECTION])

    # Autoregressive only: its internal teacher-forced numbers still go to that section,
    # and are not duplicated under the autoregressive one.
    payloads = train.eval_log_payloads(None, _autoregressive_payload())
    sections = _sections(payloads)
    assert set(sections) == {train.TEACHER_FORCED_SECTION, train.AUTOREGRESSIVE_SECTION}
    assert LOSS_KEYS <= set(sections[train.TEACHER_FORCED_SECTION])
    assert set(sections[train.AUTOREGRESSIVE_SECTION]) == ANSWER_KEYS
    assert set(sections[train.TEACHER_FORCED_SECTION]) == LOSS_KEYS


def test_teacher_forced_only_does_not_emit_an_autoregressive_section():
    payloads = train.eval_log_payloads(_teacher_payload(), None)
    sections = _sections(payloads)
    assert set(sections) == {train.TEACHER_FORCED_SECTION}
    assert set(sections[train.TEACHER_FORCED_SECTION]) == ANSWER_KEYS | LOSS_KEYS


def test_train_only_losses_join_the_teacher_forced_section():
    """ae_loss/distill_loss exist only during training, so they must be carried in explicitly."""
    teacher = {key: 0.5 for key in ANSWER_KEYS}
    teacher.update({"loss": 1.5, "qa_loss": 1.4, "reconstruction_loss": 1.3, "ppl": 4.0})
    train_terms = {
        "loss": 9.9, "qa_loss": 9.9, "reconstruction_loss": 9.9,
        "ae_loss": 1.25, "distill_loss": 0.5,
    }
    payloads = train.eval_log_payloads(teacher, None, train_terms)
    sections = _sections(payloads)
    assert set(sections) == {train.TEACHER_FORCED_SECTION}
    teacher_section = sections[train.TEACHER_FORCED_SECTION]
    assert teacher_section["ae_loss"] == 1.25
    assert teacher_section["distill_loss"] == 0.5
    # the evaluation pass stays authoritative for the numbers it computed itself
    assert teacher_section["loss"] == 1.5
    assert teacher_section["qa_loss"] == 1.4
    assert teacher_section["reconstruction_loss"] == 1.3
    # training-only terms never leak into the autoregressive section, and an evaluation pass
    # that did compute a key keeps its own value (the real evaluator produces qa_loss/ppl/
    # reconstruction_loss but never ae_loss/distill_loss).
    autoregressive = {key: 0.7 for key in ANSWER_KEYS}
    autoregressive.update({"loss": 1.9, "qa_loss": 1.9, "reconstruction_loss": 1.9, "ppl": 5.0})
    payloads = train.eval_log_payloads(None, autoregressive, train_terms)
    sections = _sections(payloads)
    assert set(sections) == {train.TEACHER_FORCED_SECTION, train.AUTOREGRESSIVE_SECTION}
    teacher_section = sections[train.TEACHER_FORCED_SECTION]
    assert teacher_section["ae_loss"] == 1.25
    assert teacher_section["distill_loss"] == 0.5
    assert teacher_section["loss"] == 1.9, "the evaluation pass must keep its own numbers"
    assert set(sections[train.AUTOREGRESSIVE_SECTION]) == ANSWER_KEYS


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
