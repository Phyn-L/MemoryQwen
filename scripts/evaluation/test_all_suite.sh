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
OUTPUT_DIR="${OUTPUT_DIR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/all_suite}"
DRY_RUN="${DRY_RUN:-0}"

export NUM_PROCESSES
args=(
  --machine "$MACHINE"
  --data-root "$DATA_ROOT"
  --datasets $DATASETS
  --memory-bs "$MEMORY_BS"
)
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
