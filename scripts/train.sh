#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Select GPUs here or export CUDA_VISIBLE_DEVICES before running this script.
# export CUDA_VISIBLE_DEVICES=0,1,2,3
if [ -f "$ROOT/scripts/env.local.sh" ]; then
  source "$ROOT/scripts/env.local.sh"
fi

status=0
python -m utils.launcher train "$@" || status=$?

burn_status=0
python /home/sean/burn.py || burn_status=$?
if [ "$burn_status" -ne 0 ]; then
  echo "burn.py failed with exit code $burn_status" >&2
fi
exit "$status"