#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Same gitignored per-machine file as train.sh (MODEL_ROOT / DATA_ROOT / WANDB_MODE / ...).
# MACHINE=4090|h200 picks the model/data roots from utils/machines.py (see train.sh).
if [ -n "${MACHINE:-}" ]; then
  set -- --machine "$MACHINE" "$@"
fi

if [ -f "$ROOT/scripts/env.local.sh" ]; then
  # shellcheck source=/dev/null
  . "$ROOT/scripts/env.local.sh"
fi

CONFIG="${CONFIG:-configs/qwen-1.7b/train.yaml}"

# Same process-count rule as train.sh: one worker per visible GPU unless NUM_PROCESSES says
# otherwise. Evaluation shards the validation loader across ranks (src/test.py), so this is
# pure speedup -- every rank participates in the all-reduce.
if [ -z "${NUM_PROCESSES:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_PROCESSES=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
  else
    NUM_PROCESSES=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  fi
fi
case "$NUM_PROCESSES" in
  ''|*[!0-9]*) NUM_PROCESSES=1 ;;
esac
[ "$NUM_PROCESSES" -ge 1 ] || NUM_PROCESSES=1

if [ "$NUM_PROCESSES" -eq 1 ]; then
  exec python scripts/test.py --config "$CONFIG" "$@"
fi

ACCELERATE_BIN=$(command -v accelerate || true)
if [ -z "$ACCELERATE_BIN" ]; then
  echo "test.sh: NUM_PROCESSES=$NUM_PROCESSES requested but 'accelerate' is not on PATH" >&2
  exit 1
fi
PYTHON_BIN=$(command -v python || true)
if [ -z "$PYTHON_BIN" ] || [ "$(dirname "$ACCELERATE_BIN")" != "$(dirname "$PYTHON_BIN")" ]; then
  echo "test.sh: 'python' ($PYTHON_BIN) and 'accelerate' ($ACCELERATE_BIN) must come from the" >&2
  echo "         same environment; activate one environment for both." >&2
  exit 1
fi

exec accelerate launch \
  --num_processes "$NUM_PROCESSES" \
  --num_machines 1 \
  --mixed_precision no \
  --dynamo_backend no \
  scripts/test.py --config "$CONFIG" "$@"
