from pathlib import Path
from utils.config import TrainConfig

ROOT = Path(__file__).resolve().parents[1]


def test_ablation_configs_change_one_factor():
    base = TrainConfig.from_file(ROOT / 'configs/train_baseline.yaml').to_dict()
    for path in (ROOT / 'configs/ablations').glob('*.yaml'):
        cfg = TrainConfig.from_file(path)
        cfg.validate()
        values = cfg.to_dict()
        differences = [(section, key) for section in base if isinstance(base[section], dict)
                       for key in base[section] if values[section][key] != base[section][key]
                       and (section, key) != ('checkpoint', 'output_dir')]
        assert len(differences) == 1, (path, differences)
