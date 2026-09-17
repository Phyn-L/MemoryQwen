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
exec python -m utils.launcher test "$@"
