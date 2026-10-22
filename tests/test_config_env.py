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
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils.config import TrainConfig, expand_env, expand_env_values  # noqa: E402


@pytest.mark.parametrize("interval", [-1, True, 1.5])
def test_context_checkpoint_interval_must_be_a_non_negative_integer(interval):
    cfg = TrainConfig()
    cfg.training.gradient_checkpoint_every_n_layers = interval
    with pytest.raises(ValueError, match="training.gradient_checkpoint_every_n_layers"):
        cfg.validate()


def test_zero_context_checkpoint_interval_disables_checkpointing_in_config():
    cfg = TrainConfig()
    cfg.training.gradient_checkpoint_every_n_layers = 0
    cfg.validate()


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


MACHINE_VARIABLES = ("MACHINE", "MODEL_ROOT", "DATA_ROOT")


def _without_machine_variables():
    """Context manager: hide this machine's overrides so the defaults can be asserted.

    Without this the test only passes where the variables happen to be unset -- on the H200,
    where scripts/env.local.sh exports them, the shipped defaults are correctly overridden and
    the assertion failed. A test must not depend on the ambient environment, and the *hostname*
    is part of that environment: utils/machines.py recognises this box and would supply
    DATA_ROOT before the config default is even consulted.
    """
    saved = {name: os.environ.pop(name, None) for name in MACHINE_VARIABLES}

    class _Restore:
        def __enter__(self):
            import socket

            self.gethostname = socket.gethostname
            socket.gethostname = lambda: "no-such-machine"
            return self

        def __exit__(self, *exc):
            import socket

            socket.gethostname = self.gethostname
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            return False

    return _Restore()


#: Runnable presets: an experiment file that combines the harness and one model layer. The
#: bases themselves (`harness.yaml`, `models/*.yaml`) are checked separately below -- they are
#: intentionally incomplete, so validating them as presets would assert nothing.
BASE_ONLY = {"harness.yaml"}
TRAINING_CONFIGS = [
    path
    for path in (
        sorted((REPO / "configs").glob("*.yaml"))
    )
    if path.name not in {"test.yaml", "icl_zeroshot.yaml", *BASE_ONLY}
    and not path.name.startswith("_")
]


def test_the_training_config_set_is_not_empty():
    assert TRAINING_CONFIGS, "the default-resolution tests below would pass vacuously"


def test_per_dataset_context_cap_is_validated_and_serialized():
    cfg = TrainConfig()
    cfg.evaluation.max_eval_contexts_per_dataset = 17
    assert cfg.to_dict()["evaluation"]["max_eval_contexts_per_dataset"] == 17
    cfg.evaluation.max_eval_contexts_per_dataset = 0
    with pytest.raises(ValueError, match="max_eval_contexts_per_dataset"):
        cfg.validate()


def test_old_eval_budget_name_is_not_accepted():
    with pytest.raises(TypeError, match="max_eval_sample_per_dataset"):
        TrainConfig.from_dict(
            {"evaluation": {"max_eval_sample_per_dataset": 37}},
        )


def test_old_qa_per_context_name_is_not_accepted():
    with pytest.raises(ValueError, match="qa_per_context"):
        TrainConfig.from_dict(
            {"data": {"qa_per_context": 4}},
        )


def test_removed_recon_context_budget_only_migrates_from_saved_configs(capsys):
    raw = {"evaluation": {"recon_max_contexts": 19}}
    with pytest.raises(ValueError, match="recon_max_contexts was removed"):
        TrainConfig.from_dict(raw)


def test_cloud_defaults_resolve_without_any_environment():
    with _without_machine_variables():
        for config in TRAINING_CONFIGS:
            cfg = TrainConfig.from_file(config)
            cfg.validate()
            assert cfg.model.name_or_path.startswith("/data/lz/hf_cache/hub/models--Qwen--"), config
            # scripts/data/ writes the 1QA splits under the repo's contexts/ tree; the two
            # conversation configs name their own SHINE-derived training file.
            assert cfg.data.root == "contexts", config
            train_file = cfg.data.split_file("train")
            if cfg.data.train_conversation_file:
                # The two conversation configs name their own SHINE-derived training file;
                # the 1QA tree is still what the validation split resolves against.
                assert train_file.endswith(("ift_c1qa.jsonl", "ift_pwc.jsonl")), config
            else:
                assert train_file == "contexts/1qa/cache/train.jsonl", config
            assert cfg.data.split_file("validation") == "contexts/1qa/cache/validation.jsonl", config


def test_the_same_config_files_resolve_to_another_machine():
    """The exact H200 override, applied to every shipped config."""
    os.environ["MODEL_ROOT"] = "/home/lijie/proj2/xmu/lz"
    os.environ["DATA_ROOT"] = "/home/lijie/proj2/xmu/lz/contexts"
    try:
        for config in TRAINING_CONFIGS:
            cfg = TrainConfig.from_file(config)
            cfg.validate()
            assert cfg.model.name_or_path.startswith("/home/lijie/proj2/xmu/lz/models--Qwen--"), config
            assert "/snapshots/" in cfg.model.name_or_path, config
            assert cfg.data.root == "/home/lijie/proj2/xmu/lz/contexts", config
    finally:
        for name in MACHINE_VARIABLES:
            os.environ.pop(name, None)


def test_no_shipped_config_points_at_the_deprecated_aggregated_tree():
    """`<root>/aggregated` is the old corpus; the training configs must not resolve to it."""
    for config in TRAINING_CONFIGS + [REPO / "configs/icl_zeroshot.yaml"]:
        assert "aggregated" not in config.read_text(encoding="utf-8"), config


def test_the_icl_baseline_defaults_follow_the_same_variables():
    """The baseline script's own defaults must not be a second hardcoded copy."""
    source = (REPO / "scripts" / "evaluation" / "test_icl_suite.py").read_text(encoding="utf-8")
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


def test_the_objective_has_its_own_config_section():
    """Loss knobs must not live under `memory:` any more, and a stale YAML must say so."""
    cfg = TrainConfig.from_dict({"loss": {"qa_weight": 0.25, "max_logits_rows": 64}})
    cfg.validate()
    assert cfg.loss.qa_weight == 0.25
    assert cfg.loss.max_logits_rows == 64
    assert not hasattr(cfg.memory, "qa_weight")
    assert not hasattr(cfg.loss, "distill_weight")
    with pytest.raises(ValueError, match="loss section"):
        TrainConfig.from_dict({"memory": {"qa_weight": 1.0}})


def test_a_bare_off_readout_mode_is_accepted_as_the_string_mode():
    """PyYAML reads `readout_mode: off` as False; the config must still mean "off"."""
    import yaml

    raw = yaml.safe_load("memory:\n  readout_mode: off\n")
    assert raw["memory"]["readout_mode"] is False
    cfg = TrainConfig.from_dict(raw)
    cfg.validate()
    assert cfg.memory.readout_mode == "off"


def test_the_harness_and_model_layers_are_bases_not_presets():
    """The layering is structural: one harness file, one model file per model, thin presets."""
    import yaml

    harness = yaml.safe_load((REPO / "configs/harness.yaml").read_text(encoding="utf-8"))
    assert "model" not in harness, "the model layer belongs in configs/models/*.yaml"
    assert "memory" not in harness, "the architecture stays with the experiment"
    model_files = sorted((REPO / "configs/models").glob("*.yaml"))
    assert model_files, "expected one model layer per shipped model"
    for path in model_files:
        assert set(yaml.safe_load(path.read_text(encoding="utf-8"))) == {"model"}, path

    # ... and every preset really does combine them.
    for path in TRAINING_CONFIGS:
        cfg = TrainConfig.from_file(path)
        cfg.validate()
        assert cfg.memory.memory_length > 0, path
        assert "Qwen" in cfg.model.name_or_path, path

    # The model layer a preset names is the one that wins.
    four_b = TrainConfig.from_file(REPO / "configs/train_baseline_qwen3_4b.yaml")
    assert "Qwen3-4B" in four_b.model.name_or_path
    eight_b = TrainConfig.from_file(REPO / "configs/train_baseline_qwen3_8b.yaml")
    assert "Qwen3-8B" in eight_b.model.name_or_path
    baseline = TrainConfig.from_file(REPO / "configs/train_baseline.yaml")
    assert "Qwen3-1.7B" in baseline.model.name_or_path
    # An experiment file changes the architecture, never the harness defaults it does not name.
    assert four_b.evaluation.max_new_tokens == 32 and baseline.evaluation.max_new_tokens == 64
    assert four_b.logging.log_every == baseline.logging.log_every == harness["logging"]["log_every"]


def test_saved_config_with_removed_fields_fails():
    full = TrainConfig.from_file(REPO / "configs/train_baseline.yaml").to_dict()
    full["data"]["validation_source_file"] = "/old/val.jsonl"
    full["data"]["validation_max_samples_per_dataset"] = {"squad": 4}

    with pytest.raises(ValueError, match="unknown data field"):
        TrainConfig.from_dict(full)
