#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Fail before anything expensive if this is not a complete checkout. A stray copy of scripts/
# (for example <data-dir>/scripts one level above the repo, which is where this was once run
# from) has no utils/ or src/ next to it, so `accelerate launch` starts one rank per GPU and
# every one of them dies with "No module named 'utils'" behind a wall of elastic output.
for required in utils src configs pyproject.toml; do
  if [ ! -e "$ROOT/$required" ]; then
    echo "train.sh: $ROOT has no '$required' -- this is not a complete checkout." >&2
    echo "train.sh: use the checkout's own script:  cd <repo> && bash scripts/train.sh" >&2
    exit 1
  fi
done

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
# CONFIG may be absolute (a scratch config outside the repo) or relative to the checkout,
# which is what `cd "$ROOT"` above makes the caller's relative paths mean.
case "$CONFIG" in
  /*) CONFIG_PATH="$CONFIG" ;;
  *)  CONFIG_PATH="$ROOT/$CONFIG" ;;
esac
if [ ! -f "$CONFIG_PATH" ]; then
  echo "train.sh: config '$CONFIG' not found (looked at $CONFIG_PATH)" >&2
  exit 1
fi

# `python` must be the project environment. A shell whose PATH puts another conda first (this
# node has a Python 3.14 miniconda ahead of the project env) yields a `python` without torch,
# and the failure surfaces much later as an import error inside every rank.
PYTHON_BIN=$(command -v python || true)
if [ -z "$PYTHON_BIN" ]; then
  echo "train.sh: no 'python' on PATH" >&2
  exit 1
fi
if ! python -c "import torch" >/dev/null 2>&1; then
  echo "train.sh: 'python' resolves to $PYTHON_BIN, which cannot import torch." >&2
  echo "train.sh: activate the project environment first (conda activate <env>), or fix PATH." >&2
  exit 1
fi

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
  exec python scripts/train.py --config "$CONFIG_PATH" "$@"
fi

# Multi-GPU requires a launcher: with plain `python`, Accelerator() reports
# distributed_type=NO / num_processes=1 and only cuda:0 is touched.
ACCELERATE_BIN=$(command -v accelerate || true)
if [ -z "$ACCELERATE_BIN" ]; then
  echo "train.sh: NUM_PROCESSES=$NUM_PROCESSES requested but 'accelerate' is not on PATH" >&2
  exit 1
fi
# A launcher from another environment runs that environment's interpreter, so the ranks would
# ignore the `python` checked above. Comparing the two bin directories catches it cheaply.
if [ "$(dirname "$ACCELERATE_BIN")" != "$(dirname "$PYTHON_BIN")" ]; then
  echo "train.sh: 'python' ($PYTHON_BIN) and 'accelerate' ($ACCELERATE_BIN) come from" >&2
  echo "train.sh: different environments; activate one environment for both." >&2
  exit 1
fi

exec accelerate launch \
  --num_processes "$NUM_PROCESSES" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  scripts/train.py --config "$CONFIG_PATH" "$@"
