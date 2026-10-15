#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Phase-3 AR candidate screening; independent training from the pretrained backbone.
# Non-loss settings (including seed) are inherited from --base-config.
BASE_CONFIG=${BASE_CONFIG:-configs/train_baseline.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/ab_loss_terms}
MACHINE=${MACHINE:-4090}
NUM_PROCESSES=${NUM_PROCESSES:-}
DRY_RUN=0

RUNS=(
    distill03_entropy_true
)

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
  [[ "$NUM_PROCESSES" =~ ^[1-9][0-9]*$ ]] || {
    echo "--num-processes must be a positive integer" >&2
    exit 2
  }
  export NUM_PROCESSES
else
  unset NUM_PROCESSES
fi

OUTPUT_ROOT="$OUTPUT_ROOT/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$OUTPUT_ROOT/configs"

BASE_CONFIG="$BASE_CONFIG" OUTPUT_ROOT="$OUTPUT_ROOT" MACHINE="$MACHINE" \
  python - <<'PY'
import copy
import os
from pathlib import Path

import yaml

from utils.config import TrainConfig

# Objective knobs live in `loss:`; everything else is architecture (`memory:`).
LOSS_KEYS = {
    "qa_weight", "recon_weight", "distill_weight", "distill_temperature",
    "distill_positions", "distill_entropy_weight", "distill_topk", "max_logits_rows",
}

base_path = Path(os.environ["BASE_CONFIG"])
out_root = Path(os.environ["OUTPUT_ROOT"])
base = TrainConfig.from_file(base_path).to_dict()

# Phase 3: shortlist AR candidates and pair entropy settings within each background.
# QA is always active; embedding reconstruction is always disabled.
# T/C use weight 1 as in phases 1/2. D uses 0.3 or the lighter 0.1.
base.setdefault("loss", {}).update(
    qa_weight=1.0,
    distill_weight=0.0,
    distill_entropy_weight=False,
    distill_positions=128,
    distill_temperature=1.0,
    distill_topk=0,
)
runs = {
    "distill03_entropy_true": {"distill_weight": 0.3, "distill_entropy_weight": True},
}

for name, overrides in runs.items():
    config = copy.deepcopy(base)
    loss_overrides = {k: v for k, v in overrides.items() if k in LOSS_KEYS}
    memory_overrides = {k: v for k, v in overrides.items() if k not in LOSS_KEYS}
    for section, values in (("loss", loss_overrides), ("memory", memory_overrides)):
        if values:
            config.setdefault(section, {}).update(values)
    config.setdefault("checkpoint", {})["output_dir"] = str(out_root / name)
    logging = config.setdefault("logging", {})
    logging["swanlab_run_name"] = f"{out_root.name}_{name}"
    logging["swanlab_run_id"] = None
    config["machine"] = os.environ["MACHINE"]

    TrainConfig.from_dict(config).validate()
    path = out_root / "configs" / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(path)
PY

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
