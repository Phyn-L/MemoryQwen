#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

NUM_GPUS=${NUM_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}
if (( NUM_GPUS > 1 )); then
  torchrun --standalone --nproc-per-node="$NUM_GPUS" scripts/test_icl_baseline.py "$@"
else
  python scripts/test_icl_baseline.py "$@"
fi
