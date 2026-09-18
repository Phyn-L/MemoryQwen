#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Serial ablation at one fixed operating point:
#   context=1024, memory=64, batch=8, causal slots, reader on.
# The three ablations change exactly one factor:
#   bidirectional: causal -> bidirectional slot attention
#   reader-off:    readout_length 8 -> 0
#   lora-off:      target_modules all -> [] (no adapters; rank is unchanged)
BASE_CONFIG=${BASE_CONFIG:-configs/4090/qwen-1.7b/reader/train_reader-on_ctx1024_m64.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/ab_slot_attention}
MACHINE=${MACHINE:-4090}
NUM_PROCESSES=${NUM_PROCESSES:-}
DRY_RUN=0
# The causal + reader-on + LoRA baseline has already been trained.  Only generate
# and run the three requested single-factor ablations below.
RUNS=(bidirectional reader-off lora-off)

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

required = {
    ("data", "max_context_tokens"): 1024,
    ("memory", "memory_length"): 64,
    ("memory", "slot_attention"): "causal",
    ("memory", "readout_length"): 8,
    ("training", "batch_size"): 8,
}
for (section, key), expected in required.items():
    actual = base.get(section, {}).get(key)
    if actual != expected:
        raise SystemExit(f"base config must set {section}.{key}={expected!r}, got {actual!r}")

runs = {
    "bidirectional": {"memory": {"slot_attention": "bidirectional"}},
    "reader-off": {"memory": {"readout_length": 0}},
    "lora-off": {"model": {"target_modules": []}},
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
