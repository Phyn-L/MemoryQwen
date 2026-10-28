from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.model import load_model as load_project_model
import scripts.train as project_train
Compressor500xModel = importlib.import_module(
    "baselines.500xCompressor.model"
).Compressor500xModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--resume")
    parser.add_argument("--machine")
    parser.add_argument("--data-root")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], "--config", args.config]
    for key in ("resume", "machine", "data-root"):
        value = getattr(args, key.replace("-", "_"))
        if value:
            sys.argv.extend((f"--{key}", value))
    sys.argv.extend(remaining)

    def load(cfg):
        tokenizer, model = load_project_model(cfg)
        return tokenizer, Compressor500xModel.adapt(model)

    project_train.load_model = load
    project_train.main()


if __name__ == "__main__":
    main()
