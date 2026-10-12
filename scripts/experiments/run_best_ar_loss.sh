#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Best current single-seed AR candidate: distillation only.
# Observed in 09_distill_only: AR F1=0.6814, ROUGE-L=0.6776, EM=0.5566.
BASE_CONFIG=${BASE_CONFIG:-configs/train_baseline.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/ab_best_ar_loss}
MACHINE=${MACHINE:-4090}
NUM_PROCESSES=${NUM_PROCESSES:-}
DRY_RUN=0
RUN_NAME=${RUN_NAME:-10_best_ar_distill_only}

usage() {
  echo "Usage: $0 [--base-config PATH] [--output-root PATH] [--machine NAME] [--num-processes N] [--run-name NAME] [--dry-run]"
}

while (($#)); do
  case "$1" in
    --base-config) BASE_CONFIG=$2; shift 2;;
    --output-root) OUTPUT_ROOT=$2; shift 2;;
    --machine) MACHINE=$2; shift 2;;
    --num-processes) NUM_PROCESSES=$2; shift 2;;
    --run-name) RUN_NAME=$2; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done

[[ -f "$BASE_CONFIG" ]] || { echo "base config not found: $BASE_CONFIG" >&2; exit 2; }
[[ "$RUN_NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  echo "--run-name may contain only letters, numbers, '.', '_' and '-'" >&2
  exit 2
}
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

BASE_CONFIG="$BASE_CONFIG" OUTPUT_ROOT="$OUTPUT_ROOT" MACHINE="$MACHINE" RUN_NAME="$RUN_NAME" \
  python - <<'PY'
import copy
import os
from pathlib import Path

import yaml

from utils.config import TrainConfig

base_path = Path(os.environ["BASE_CONFIG"])
out_root = Path(os.environ["OUTPUT_ROOT"])
run_name = os.environ["RUN_NAME"]
config = copy.deepcopy(yaml.safe_load(base_path.read_text(encoding="utf-8")))

# Best observed AR configuration: QA + distribution distillation only.
config.setdefault("memory", {}).update(
    distill_weight=0.3,
    distill_entropy_weight=True,
)
config.setdefault("checkpoint", {})["output_dir"] = str(out_root / run_name)
logging = config.setdefault("logging", {})
logging["swanlab_run_name"] = f"{out_root.name}_{run_name}"
logging["swanlab_run_id"] = None
config["machine"] = os.environ["MACHINE"]

TrainConfig.from_dict(config).validate()
path = out_root / "configs" / f"{run_name}.yaml"
path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print(path)
PY

config="$OUTPUT_ROOT/configs/$RUN_NAME.yaml"
[[ -f "$config" ]] || {
  echo "generated config missing: $config" >&2
  exit 1
}

echo "[ablation] starting $RUN_NAME"
if ((DRY_RUN)); then
  printf 'bash scripts/train.sh --machine %q --config %q\n' "$MACHINE" "$config"
else
  bash scripts/train.sh --machine "$MACHINE" --config "$config" 2>&1 | tee "$OUTPUT_ROOT/$RUN_NAME.log"
  echo "[ablation] finished $RUN_NAME"
fi

echo "[ablation] output: $OUTPUT_ROOT"
