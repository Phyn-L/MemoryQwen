#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [ -f scripts/env.local.sh ]; then source scripts/env.local.sh; fi

# Edit these variables for a run. Command-line arguments appended below override them.
MACHINE="${MACHINE:-h200}"
NUM_PROCESSES="${NUM_PROCESSES:-6}"
CKPT_DIR="${CKPT_DIR:-outputs/checkpoints}"
DATA_ROOT="${DATA_ROOT:-}"
DATASETS="${DATASETS:-squad ms_marco_v1 ms_marco_v2 race hotpotqa}"
MEMORY_BS="${MEMORY_BS:-256}"
SYSTEM_PROMPT_MODE="${SYSTEM_PROMPT_MODE:-context}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-1024}"
MAX_CONTEXT_ROWS="${MAX_CONTEXT_ROWS:-10000}"
CONTEXT_SHUFFLE_SEED="${CONTEXT_SHUFFLE_SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval_ab_loss_term}"
DRY_RUN="${DRY_RUN:-0}"

has_option() {
  local option="$1" arg
  shift
  for arg in "$@"; do
    if [[ "$arg" == "$option" || "$arg" == "$option="* ]]; then
      return 0
    fi
  done
  return 1
}

export NUM_PROCESSES
args=(
  --machine "$MACHINE"
  --datasets $DATASETS
  --memory-bs "$MEMORY_BS"
  --system-prompt-mode "$SYSTEM_PROMPT_MODE"
  --max-context-tokens "$MAX_CONTEXT_TOKENS"
)
if [[ -n "$DATA_ROOT" ]]; then args+=(--data-root "$DATA_ROOT"); fi
if [[ -n "$MAX_CONTEXT_ROWS" ]]; then args+=(--max-context-rows "$MAX_CONTEXT_ROWS"); fi
args+=(--context-shuffle-seed "$CONTEXT_SHUFFLE_SEED")
if has_option --checkpoint "$@" || has_option --ckpt "$@" || has_option --ckpt-dir "$@"; then
  : # Explicit checkpoint arguments below take precedence over CKPT_DIR.
else
  args+=(--ckpt-dir "$CKPT_DIR")
fi
if ! has_option --output-dir "$@"; then
  args+=(--output-dir "$OUTPUT_DIR")
fi
if [[ "$DRY_RUN" == "1" ]] && ! has_option --dry-run "$@"; then
  args+=(--dry-run)
fi
exec python scripts/evaluation/test_all_suite.py "${args[@]}" "$@"

sleep 1m
