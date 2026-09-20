#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f scripts/env.local.sh ]]; then source scripts/env.local.sh; fi

# Edit these settings; CLI arguments override the defaults below.
MACHINE="${MACHINE:-4090}"
MODEL="${MODEL:-Qwen3-8B}"
DATASETS="${DATASETS:-squad ms_marco_v1 ms_marco_v2 hotpotqa race}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
# Leave unset to use all visible GPUs; e.g. export NUM_PROCESSES=4.
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export NUM_PROCESSES=4

read -r -a dataset_args <<< "$DATASETS"
args=(--machine "$MACHINE" --model "$MODEL" --datasets "${dataset_args[@]}"
      --bs "$BATCH_SIZE" --max-input-tokens "$MAX_INPUT_TOKENS"
      --max-new-tokens "$MAX_NEW_TOKENS")
if [[ -n "$OUTPUT_DIR" ]]; then args+=(--output-dir "$OUTPUT_DIR"); fi
exec python -m scripts.evaluation.test_icl_suite "${args[@]}" "$@"
