#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

training_status=0
run_burn() {
  local burn_status=0
  python /home/sean/burn.py || burn_status=$?
  if [ "$burn_status" -ne 0 ]; then
    echo "burn.py failed with exit code $burn_status" >&2
  fi
}
trap run_burn EXIT

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 {icae|500xCompressor} [training arguments...]" >&2
  exit 2
fi
method=$1
shift
case "$method" in
  icae) module=baselines.icae.train ;;
  500xCompressor) module=baselines.500xCompressor.train ;;
  *)
    echo "Unknown baseline '$method'; choose icae or 500xCompressor" >&2
    exit 2
    ;;
esac

gpu_count=$(python -c 'import torch; print(torch.cuda.device_count())') || exit 1
if [ "$gpu_count" -lt 1 ]; then
  echo "No visible CUDA GPUs found" >&2
  exit 1
fi

echo "Launching $method with $gpu_count GPU(s)"
accelerate launch --multi_gpu --num_processes "$gpu_count" \
  -m "$module" "$@" || training_status=$?
exit "$training_status"
