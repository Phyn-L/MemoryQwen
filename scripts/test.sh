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
exec python scripts/test.py --config "$CONFIG" "$@"
