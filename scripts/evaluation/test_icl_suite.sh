#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f scripts/env.local.sh ]]; then source scripts/env.local.sh; fi

# Edit these settings; CLI arguments override the defaults below.
MACHINE="${MACHINE:-4090}"
MODEL="${MODEL:-Qwen3-8B}"
PROMPT_MODE="${PROMPT_MODE:-instruct}"
DATASETS="${DATASETS:-squad ms_marco_v1 ms_marco_v2 hotpotqa race}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-1024}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-1024}"
MAX_CONTEXTS="${MAX_CONTEXTS:-}"
MAX_QA_SAMPLE_PER_CONTEXT="${MAX_QA_SAMPLE_PER_CONTEXT:-}"
CONTEXT_SHUFFLE_SEED="${CONTEXT_SHUFFLE_SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
# Leave unset to use all visible GPUs; e.g. export NUM_PROCESSES=4.
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export NUM_PROCESSES=4

read -r -a dataset_args <<< "$DATASETS"
args=(--machine "$MACHINE" --model "$MODEL" --datasets "${dataset_args[@]}"
      --prompt-mode "$PROMPT_MODE"
      --bs "$BATCH_SIZE" --max-input-tokens "$MAX_INPUT_TOKENS"
      --max-new-tokens "$MAX_NEW_TOKENS" --max-context-tokens "$MAX_CONTEXT_TOKENS")
if [[ -n "$MAX_CONTEXTS" ]]; then args+=(--max-contexts "$MAX_CONTEXTS"); fi
if [[ -n "$MAX_QA_SAMPLE_PER_CONTEXT" ]]; then
  args+=(--max-qa-sample-per-context "$MAX_QA_SAMPLE_PER_CONTEXT")
fi
args+=(--context-shuffle-seed "$CONTEXT_SHUFFLE_SEED")
if [[ -n "$OUTPUT_DIR" ]]; then args+=(--output-dir "$OUTPUT_DIR"); fi
exec python -m scripts.evaluation.test_icl_suite "${args[@]}" "$@"
