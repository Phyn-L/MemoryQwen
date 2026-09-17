#!/usr/bin/env bash
# Generic wrapper for the ICL baseline (RACE-oriented: 0-shot default, three pinned cards).
# For the SQuAD validation baseline of Qwen3-1.7B / Qwen3-8B in the memory project's protocol
# (full split, v1/v2/full subsets, one output directory per run, a results.md table) use
# `scripts/eval_icl_squad.sh` instead.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT=$PWD
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Same gitignored per-machine file as train.sh. Put MODEL_ROOT / DATA_ROOT / WANDB_MODE there
# instead of editing this script, so `git pull` never conflicts.
# MACHINE=4090|h200 picks the model/data roots from utils/machines.py (see train.sh).
if [ -n "${MACHINE:-}" ]; then
  set -- --machine "$MACHINE" "$@"
fi

if [ -f "$ROOT/scripts/env.local.sh" ]; then
  # shellcheck source=/dev/null
  . "$ROOT/scripts/env.local.sh"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export OMP_NUM_THREADS=1

MODEL="${MODEL:-${MODEL_ROOT:-/data/lz/hf_cache/hub}/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"
DATASETS="${DATASETS:-race}"
NUM_SHOTS="${NUM_SHOTS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/icl_baseline/qwen3-1.7b-0shot}"

# One process per visible GPU, so a machine with a different card count needs no edit here.
NUM_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
if (( NUM_GPUS == 0 )); then
  echo "没有检测到可用 GPU，请检查 CUDA_VISIBLE_DEVICES 和驱动。"
  exit 1
fi

python -m torch.distributed.run \
  --standalone \
  --nproc-per-node="$NUM_GPUS" \
  scripts/test_icl_baseline.py \
  --model "$MODEL" \
  --datasets $DATASETS \
  --num-shots "$NUM_SHOTS" \
  --batch-size "${BATCH_SIZE:-64}" \
  --max-input-tokens 8192 \
  --squad-max-new-tokens 32 \
  --race-max-new-tokens 8 \
  --dtype bfloat16 \
  --output-dir "$OUTPUT_DIR" \
  "$@"
