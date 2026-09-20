#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Independent cumulative ablations, trained from the same pretrained backbone.
BASE_CONFIG=${BASE_CONFIG:-configs/train_baseline.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/ab_cumulative_memory}
MACHINE=${MACHINE:-4090}
NUM_PROCESSES=${NUM_PROCESSES:-}
DRY_RUN=0
# Keep this list identical to the keys generated below.  A mismatch here makes
# the shell look for a YAML file that the generator never wrote.
RUNS=(04_bidirectional 05_reader_len_16 06_reader_len16_layer4)

usage() {
  echo "Usage: $0 [--base-config PATH] [--output-root PATH] [--machine NAME] [--num-processes N] [--dry-run]"
}

while (($#)); do
  case "$1" in
    --base-config) BASE_CONFIG=$2; shift 2;;
    --output-root) OUTPUT_ROOT=$2; shift 2;;
    --machine) MACHINE=$2; shift 2;;
    --num-processes) NUM_PROCESSES=$2; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done

[[ -f "$BASE_CONFIG" ]] || { echo "base config not found: $BASE_CONFIG" >&2; exit 2; }
if [[ -n "$NUM_PROCESSES" ]]; then
  [[ "$NUM_PROCESSES" =~ ^[1-9][0-9]*$ ]] || { echo "--num-processes must be a positive integer" >&2; exit 2; }
  export NUM_PROCESSES
else
  unset NUM_PROCESSES
fi
OUTPUT_ROOT="$OUTPUT_ROOT/$(date +%Y%m%d_%H%M%S)_$$"

mkdir -p "$OUTPUT_ROOT/configs"

# Use the activated environment, just like scripts/train.sh.
BASE_CONFIG="$BASE_CONFIG" OUTPUT_ROOT="$OUTPUT_ROOT" MACHINE="$MACHINE" \
  python - <<'PY'
import copy
import os
from pathlib import Path
import yaml
from utils.config import TrainConfig

base_path = Path(os.environ["BASE_CONFIG"])
out_root = Path(os.environ["OUTPUT_ROOT"])
base = yaml.safe_load(base_path.read_text(encoding="utf-8"))

# Shape and slot attention are fixed for all three independent runs.
base.setdefault("data", {})["max_context_tokens"] = 1024
base.setdefault("memory", {}).update(memory_length=64, slot_attention="causal")
resolved = TrainConfig.from_dict(base)
targets = list(resolved.model.target_modules)
if not targets:
    raise SystemExit("base config must specify non-empty model.target_modules for LoRA arms")
reader_length = resolved.memory.readout_length
if reader_length <= 0:
    raise SystemExit("base config must specify positive memory.readout_length for reader arm")
runs = {
    "04_bidirectional": {"model": {"target_modules": targets}, "memory": {"slot_attention": "bidirectional"}},
    "05_reader_len_16": {"model": {"target_modules": targets}, "memory": {"readout_length": 16}},
    "06_reader_len16_layer4": {"model": {"target_modules": targets}, "memory": {"readout_length": 16, "readout_layers": 4}},
}
for name, overrides in runs.items():
    config = copy.deepcopy(base)
    for section, values in overrides.items():
        config.setdefault(section, {}).update(values)
    config.setdefault("checkpoint", {})["output_dir"] = str(out_root / name)
    config.setdefault("logging", {})["wandb_run_name"] = f"{out_root.name}_{name}"
    config["logging"]["wandb_run_id"] = None
    config["machine"] = os.environ["MACHINE"]
    TrainConfig.from_dict(config).validate()
    path = out_root / "configs" / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(path)
PY

# Fail at the generation boundary with a useful message instead of attempting
# to launch training with a missing configuration.
for name in "${RUNS[@]}"; do
  [[ -f "$OUTPUT_ROOT/configs/$name.yaml" ]] || {
    echo "generated config missing: $OUTPUT_ROOT/configs/$name.yaml" >&2
    exit 1
  }
done

for name in "${RUNS[@]}"; do
  config="$OUTPUT_ROOT/configs/$name.yaml"
  echo "[ablation] starting $name"
  if ((DRY_RUN)); then
    printf 'bash scripts/train.sh --machine %q --config %q\n' "$MACHINE" "$config"
    continue
  fi
  bash scripts/train.sh --machine "$MACHINE" --config "$config" 2>&1 | tee "$OUTPUT_ROOT/$name.log"
  echo "[ablation] finished $name"
done

echo "[ablation] all runs finished; outputs: $OUTPUT_ROOT"
