#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [ -f scripts/env.local.sh ]; then source scripts/env.local.sh; fi

# Edit these variables for a run. Command-line arguments appended below override them.
MACHINE="${MACHINE:-4090}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/checkpoints}"
DATA_ROOT="${DATA_ROOT:-/data/lz/contexts/aggregated}"
DATASETS="${DATASETS:-squad ms_marco_v1 ms_marco_v2 hotpotqa race}"
MEMORY_BS="${MEMORY_BS:-1}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-1024}"
MAX_CONTEXT_ROWS="${MAX_CONTEXT_ROWS:-}"
QA_PER_CONTEXT="${QA_PER_CONTEXT:-}"
CONTEXT_SHUFFLE_SEED="${CONTEXT_SHUFFLE_SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/all_suite}"
DRY_RUN="${DRY_RUN:-0}"

export NUM_PROCESSES
args=(
  --machine "$MACHINE"
  --data-root "$DATA_ROOT"
  --datasets $DATASETS
  --memory-bs "$MEMORY_BS"
  --max-context-tokens "$MAX_CONTEXT_TOKENS"
)
if [[ -n "$MAX_CONTEXT_ROWS" ]]; then args+=(--max-context-rows "$MAX_CONTEXT_ROWS"); fi
if [[ -n "$QA_PER_CONTEXT" ]]; then args+=(--qa-per-context "$QA_PER_CONTEXT"); fi
args+=(--context-shuffle-seed "$CONTEXT_SHUFFLE_SEED")
if [[ "$*" == *"--checkpoint "* || "$*" == *"--ckpt "* || "$*" == *"--checkpoint="* || "$*" == *"--ckpt="* ]]; then
  : # explicit checkpoint arguments below take precedence over CHECKPOINT_DIR
else
  args+=(--checkpoint-dir "$CHECKPOINT_DIR")
fi
if [[ -n "$OUTPUT_DIR" && "$*" != *"--output-dir"* && "$*" != *"--output-root"* ]]; then
  args+=(--output-dir "$OUTPUT_DIR")
elif [[ "$*" != *"--output-dir"* && "$*" != *"--output-root"* ]]; then
  args+=(--output-root "$OUTPUT_ROOT")
fi
if [[ "$DRY_RUN" == "1" && "$*" != *"--dry-run"* ]]; then
  args+=(--dry-run)
fi
exec python scripts/evaluation/test_all_suite.py "${args[@]}" "$@"
