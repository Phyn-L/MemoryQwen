#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Qwen3 v2 writer ablations.  This script only varies components introduced by the
# context-forward + ContextMemoryResampler architecture.  The old reader modes,
# token-reconstruction/distillation objectives, and memory-length experiments are
# deliberately held fixed and are not treated as v2 factors here.
BASE_CONFIG=${BASE_CONFIG:-configs/train_baseline_qwen3_8b.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/ab_v2_writer_qwen3_8b}
MACHINE=${MACHINE:-h200}
NUM_PROCESSES=${NUM_PROCESSES:-8}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-250000}
WARMUP_STEPS=${WARMUP_STEPS:-800}
AR_EVERY=${AR_EVERY:-2000}
DRY_RUN=0

# Keep the list in the same order as the generated configs.  The first arm is the
# current v2 baseline; the remaining arms test one writer/training factor at a time.
RUNS=(
  01_baseline_mix50
  02_layers_last3
  03_layers_last5
  04_resampler_1layer
  05_resampler_4layer
  06_width128
  07_width512
  08_mix10
  09_mix90
)

usage() {
  echo "Usage: $0 [--base-config PATH] [--output-root PATH] [--machine NAME] [--num-processes N] [--dry-run]"
  echo "       TRAIN_MAX_SAMPLES=250000 (or null for the full corpus), WARMUP_STEPS=800, AR_EVERY=2000"
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
[[ "$NUM_PROCESSES" =~ ^[1-9][0-9]*$ ]] || {
  echo "--num-processes must be a positive integer" >&2
  exit 2
}
export NUM_PROCESSES
if [[ "$TRAIN_MAX_SAMPLES" != "null" && ! "$TRAIN_MAX_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAIN_MAX_SAMPLES must be a positive integer or null" >&2
  exit 2
fi
[[ "$WARMUP_STEPS" =~ ^[1-9][0-9]*$ && "$AR_EVERY" =~ ^[1-9][0-9]*$ ]] || {
  echo "WARMUP_STEPS and AR_EVERY must be positive integers" >&2
  exit 2
}

OUTPUT_ROOT="$OUTPUT_ROOT/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$OUTPUT_ROOT/configs" "$OUTPUT_ROOT/logs"

BASE_CONFIG="$BASE_CONFIG" OUTPUT_ROOT="$OUTPUT_ROOT" MACHINE="$MACHINE" TRAIN_MAX_SAMPLES="$TRAIN_MAX_SAMPLES" WARMUP_STEPS="$WARMUP_STEPS" AR_EVERY="$AR_EVERY" \
  python - <<'PY'
import copy
import os
from pathlib import Path

import yaml

from utils.config import TrainConfig

base_path = Path(os.environ["BASE_CONFIG"])
out_root = Path(os.environ["OUTPUT_ROOT"])
base = TrainConfig.from_file(base_path).to_dict()

# Resource contract for 8x H200. Four contexts per rank gives global batch 32.
# The new Shine JSONL is much larger than the old aggregated corpus, so the default
# screening budget uses 250k contexts (the historical A/B scale). Set
# TRAIN_MAX_SAMPLES=null for a full-corpus run after selecting a winner.
base.setdefault("data", {}).update(
    max_context_tokens=1024,
    max_input_tokens=40960,
    max_answer_tokens=64,
    train_tasks="mix",
    recon_ratio=0.5,
    recon_schedule="constant",
    recon_schedule_curve="linear",
    recon_ratio_start=0.5,
    recon_ratio_end=0.5,
    recon_ratio_min=0.1,
    qa_per_context=4,
)
base["data"]["train_max_samples"] = (
    None if os.environ["TRAIN_MAX_SAMPLES"] == "null"
    else int(os.environ["TRAIN_MAX_SAMPLES"])
)
base.setdefault("training", {}).update(batch_size=4, epochs=1, deterministic=False)
base.setdefault("scheduler", {})["warmup_steps"] = int(os.environ["WARMUP_STEPS"])
base.setdefault("evaluation", {}).update(
    # 250k contexts × two mixed tasks / global batch 32 ≈ 15.6k steps.
    # AR every 2000 gives eight checkpoints plus the mandatory final evaluation.
    autoregressive_every=int(os.environ["AR_EVERY"]),
    max_eval_sample_per_dataset=1024,
    qa_batch_size=4,
    max_new_tokens=32,
)
base.setdefault("logging", {}).update(log_every=25)

# These are legacy auxiliary branches.  Keep them off in every v2 arm so the
# comparison measures the new writer and task mixture rather than a second loss.
base.setdefault("memory", {}).update(
    readout_length=8,
    context_layer_indices=[-1],
    memory_resampler_layers=2,
    memory_resampler_heads=4,
    memory_resampler_hidden_size=256,
)
base.setdefault("loss", {}).update(
    distill_weight=0.0,
)

runs = {
    # Fixed reference: one final-layer representation, 2-layer/256-wide writer,
    # and equal QA/Recon supervision.
    "01_baseline_mix50": {},
    # Does exposing intermediate encoder states help the cross-attention writer?
    "02_layers_last3": {"memory": {"context_layer_indices": [-1, -8, -16]}},
    "03_layers_last5": {"memory": {"context_layer_indices": [-1, -4, -8, -12, -16]}},
    # Does writer depth, at fixed bottleneck width, improve compression?
    "04_resampler_1layer": {"memory": {"memory_resampler_layers": 1}},
    "05_resampler_4layer": {"memory": {"memory_resampler_layers": 4}},
    # Capacity sweep for the new cross-attention bottleneck.  Heads stay at four
    # so the principal change is width rather than a simultaneous head-count change.
    "06_width128": {"memory": {"memory_resampler_hidden_size": 128}},
    "07_width512": {"memory": {"memory_resampler_hidden_size": 512}},
    # The dataset expansion is identical in all mix arms; only the scheduled
    # QA/Recon loss weights change, so the number of optimizer steps is comparable.
    "08_mix10": {
        "data": {"recon_ratio": 0.1, "recon_ratio_start": 0.1, "recon_ratio_end": 0.1}
    },
    "09_mix90": {
        "data": {"recon_ratio": 0.9, "recon_ratio_start": 0.9, "recon_ratio_end": 0.9}
    },
}

for name, overrides in runs.items():
    config = copy.deepcopy(base)
    for section, values in overrides.items():
        config.setdefault(section, {}).update(values)
    config.setdefault("checkpoint", {})["output_dir"] = str(out_root)
    config.setdefault("logging", {})["aim_run_name"] = name
    config["logging"]["aim_run_id"] = None
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
  bash scripts/train.sh --machine "$MACHINE" --config "$config" 2>&1 | tee "$OUTPUT_ROOT/logs/$name.log"
  echo "[ablation] finished $name"
done

echo "[ablation] all runs finished; outputs: $OUTPUT_ROOT"
