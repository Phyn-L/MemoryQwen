#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Phase-7 token reconstruction position comparison; independent training runs.
# Non-loss settings (including seed) are inherited from --base-config.
BASE_CONFIG=${BASE_CONFIG:-configs/train_baseline_qwen3_8b.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/ab_loss_terms/Qwen3_8B}
MACHINE=${MACHINE:-h200}
NUM_PROCESSES=${NUM_PROCESSES:-6}
DRY_RUN=0

RUNS=(
    01token_recon128-distill03_false_256
    02token_recon128-distill03_true_256
    03token_recon128-distill01_false_256
    04token_recon128-distill01_true_256
    05token_recon256-distill03_false_256
    06token_recon128-distill05_false_256
    07token_recon128-distill10_false_256
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

mkdir -p "$OUTPUT_ROOT/configs"

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

# Phase 7: compare 128 vs 256 token reconstruction positions.
# QA and token reconstruction use weight 1; distillation uses weight 0.3
# with 256 positions and entropy weighting disabled. Other objectives are off.
base.setdefault("memory", {}).update(
    qa_weight=1.0,
    embedding_recon_weight=0.0,
    token_recon_weight=1.0,
    causal_recon_weight=0.0,
    distill_weight=0.0,
    distill_entropy_weight=False,
    token_recon_positions=128,
    causal_recon_positions=128,
    distill_positions=128,
    distill_temperature=1.0,
    distill_topk=0,
)
runs = {
    "01token_recon128-distill03_false_256":{"distill_weight": 0.3, "distill_entropy_weight": False, "distill_positions": 256},
    "02token_recon128-distill03_true_256":{"distill_weight": 0.3, "distill_entropy_weight": True, "distill_positions": 256},
    "03token_recon128-distill01_false_256":{"distill_weight": 0.1, "distill_entropy_weight": False, "distill_positions": 256},
    "04token_recon128-distill01_true_256":{"distill_weight": 0.1, "distill_entropy_weight": True, "distill_positions": 256},
    "05token_recon256-distill03_false_256":{"distill_weight": 0.3, "distill_entropy_weight": False, "distill_positions": 256, "token_recon_positions": 256},
    "06token_recon128-distill05_false_256":{"distill_weight": 0.5, "distill_entropy_weight": False, "distill_positions": 256},
    "07token_recon128-distill10_false_256":{"distill_weight": 1.0, "distill_entropy_weight": False, "distill_positions": 256},
}

for name, overrides in runs.items():
    config = copy.deepcopy(base)
    memory = config.setdefault("memory", {})
    memory.update(overrides)
    config.setdefault("checkpoint", {})["output_dir"] = str(out_root)
    logging = config.setdefault("logging", {})
    logging["wandb_run_name"] = name
    logging["wandb_run_id"] = None
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

python /home/sean/burn.py