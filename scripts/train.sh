#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
if command -v accelerate >/dev/null 2>&1; then
  accelerate launch \
    --num_processes 4 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    scripts/train.py --config configs/qwen-1.7b/train.yaml "$@"
else
    python scripts/train.py --config configs/qwen-1.7b/train.yaml "$@"
fi
