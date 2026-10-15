#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Qwen3-1.7B v2 writer ablations.  This script targets the current
# context-forward + ContextMemoryResampler architecture.  Legacy reader modes,
# fixed and are intentionally not treated as v2 factors.
BASE_CONFIG=${BASE_CONFIG:-configs/train_baseline.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/ab_v2_writer_qwen3_1_7b}
MACHINE=${MACHINE:-h200}
NUM_PROCESSES=${NUM_PROCESSES:-8}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-250000}
WARMUP_STEPS=${WARMUP_STEPS:-800}
AR_EVERY=${AR_EVERY:-2000}
DRY_RUN=0

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

BASE_CONFIG="$BASE_CONFIG" OUTPUT_ROOT="$OUTPUT_ROOT" MACHINE="$MACHINE" \
TRAIN_MAX_SAMPLES="$TRAIN_MAX_SAMPLES" WARMUP_STEPS="$WARMUP_STEPS" AR_EVERY="$AR_EVERY" \
  python - <<'PY'
import copy
import os
from pathlib import Path

import yaml

from utils.config import TrainConfig

base_path = Path(os.environ["BASE_CONFIG"])
out_root = Path(os.environ["OUTPUT_ROOT"])
base = TrainConfig.from_file(base_path).to_dict()

# 8x H200 resource contract for Qwen3-1.7B.  The per-rank batch is deliberately
# conservative for the longest 4096-token contexts; global batch is 4 * 8 = 32.
# The new conversation JSONL is much larger than the old aggregated corpus, so
# screening defaults to 250k contexts.  Set TRAIN_MAX_SAMPLES=null after selecting
# a candidate to run one final full-corpus experiment.
base.setdefault("data", {}).update(
    max_context_tokens=4096,
    max_input_tokens=40960,
    max_question_tokens=96,
    max_conversation_tokens=768,
    max_answer_tokens=128,
    train_tasks="mix",
    recon_ratio=0.5,
    recon_schedule="constant",
    recon_schedule_curve="linear",
    recon_ratio_start=0.5,
    recon_ratio_end=0.5,
    recon_ratio_min=0.1,
    max_qa_sample_per_context=4,
)
base["data"]["train_max_samples"] = (
    None if os.environ["TRAIN_MAX_SAMPLES"] == "null"
    else int(os.environ["TRAIN_MAX_SAMPLES"])
)
base.setdefault("training", {}).update(batch_size=4, epochs=1, deterministic=False)
base.setdefault("scheduler", {})["warmup_steps"] = int(os.environ["WARMUP_STEPS"])
base.setdefault("evaluation", {}).update(
    # 250k contexts x two mixed tasks / global batch 32 ~= 15.6k steps.
    # AR every 2000 gives several checkpoints plus the mandatory final evaluation.
    autoregressive_every=int(os.environ["AR_EVERY"]),
    max_eval_contexts_per_dataset=1024,
    qa_batch_size=8,
    max_new_tokens=32,
)
base.setdefault("logging", {}).update(log_every=25)

# Keep the old auxiliary branches off in every v2 arm.  The only varied factors
# below are the new context writer and the QA/Recon supervision mixture.
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
    "01_baseline_mix50": {},
    "02_layers_last3": {"memory": {"context_layer_indices": [-1, -8, -16]}},
    "03_layers_last5": {"memory": {"context_layer_indices": [-1, -4, -8, -12, -16]}},
    "04_resampler_1layer": {"memory": {"memory_resampler_layers": 1}},
    "05_resampler_4layer": {"memory": {"memory_resampler_layers": 4}},
    "06_width128": {"memory": {"memory_resampler_hidden_size": 128}},
    "07_width512": {"memory": {"memory_resampler_hidden_size": 512}},
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
    config.setdefault("logging", {})["swanlab_run_name"] = name
    config["logging"]["swanlab_run_id"] = None
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
