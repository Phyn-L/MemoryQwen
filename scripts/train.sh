#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Per-machine settings live in a gitignored file so a cluster with different paths never has
# to edit tracked files -- and therefore never conflicts on `git pull`. Typical content:
#   export MODEL_ROOT=/home/lijie/proj2/xmu/lz
#   export DATA_ROOT=/home/lijie/proj2/xmu/lz/aggregated
#   export WANDB_MODE=offline
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
# See the README section "Running on another machine".
if [ -f "$ROOT/scripts/env.local.sh" ]; then
  # shellcheck source=/dev/null
  . "$ROOT/scripts/env.local.sh"
fi

# The two OOM runs told us to set this; set it in the entry point instead of by hand.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG="${CONFIG:-configs/qwen-1.7b/train.yaml}"

# One worker per visible GPU unless NUM_PROCESSES says otherwise. A hardcoded device list is
# how a run silently ends up on the wrong number of cards, and via CUDA_VISIBLE_DEVICES it
# also hides the rest of the node.
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
  exec python scripts/train.py --config "$CONFIG" "$@"
fi

# Multi-GPU requires a launcher: with plain `python`, Accelerator() reports
# distributed_type=NO / num_processes=1 and only cuda:0 is touched. Fail loudly instead of
# silently training on one card.
command -v accelerate >/dev/null 2>&1 || {
  echo "train.sh: NUM_PROCESSES=$NUM_PROCESSES requested but 'accelerate' is not on PATH" >&2
  exit 1
}

exec accelerate launch \
  --num_processes "$NUM_PROCESSES" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  scripts/train.py --config "$CONFIG" "$@"
