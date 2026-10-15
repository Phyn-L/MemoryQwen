from pathlib import Path

from utils.config import TrainConfig

ROOT = Path(__file__).resolve().parents[1]
ABLATIONS = ROOT / "configs/ablations"
# Settings every ablation shares (evaluation cadence, batch size, warm-up, validation
# subset). They live in one base file so the one-factor assertion below can actually hold.
HARNESS = ABLATIONS / "_ablation_common.yaml"


def ablation_files():
    return [path for path in sorted(ABLATIONS.glob("*.yaml")) if not path.name.startswith("_")]


def differences(values, reference):
    return [
        (section, key)
        for section in reference
        if isinstance(reference[section], dict)
        for key in reference[section]
        if values[section][key] != reference[section][key]
        and (section, key) != ("checkpoint", "output_dir")
    ]


def test_the_shared_harness_is_a_base_and_still_resolves():
    cfg = TrainConfig.from_file(HARNESS)
    cfg.validate()
    assert cfg.data.validation_datasets == ("squad",)
    assert cfg.evaluation.autoregressive_every == 400
    # Inherited from train_baseline.yaml through the harness.
    assert cfg.model.name_or_path.startswith("/data/lz/hf_cache/hub/models--Qwen--")


def test_ablation_configs_change_one_factor():
    """Each ablation differs from the shared harness by exactly its own variable.

    The comparison baseline is the harness, not `train_baseline.yaml`: the harness carries the
    settings every ablation shares, so against the raw baseline those would count as factors
    and the assertion could only be satisfied by copying them into every file -- which is what
    it used to do, leaving the test red once the copies drifted.
    """
    reference = TrainConfig.from_file(HARNESS).to_dict()
    files = ablation_files()
    assert files, "no ablation configs found"
    for path in files:
        cfg = TrainConfig.from_file(path)
        cfg.validate()
        found = differences(cfg.to_dict(), reference)
        assert len(found) == 1, (path, found)


def test_every_ablation_config_is_covered():
    """A new ablation file must show up in the one-factor check rather than be skipped."""
    names = {path.stem for path in ablation_files()}
    assert names == {
        "context_length",
        "lora_off",
        "memory_length",
        "multilayer",
        "reader_embedding",
        "reader_off",
    }
