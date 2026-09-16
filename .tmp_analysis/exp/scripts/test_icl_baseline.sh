#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2
export OMP_NUM_THREADS=1

NUM_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
if (( NUM_GPUS == 0 )); then
  echo "没有检测到可用 GPU，请检查 CUDA_VISIBLE_DEVICES 和驱动。"
  exit 1
fi

python -m torch.distributed.run \
  --standalone \
  --nproc-per-node="$NUM_GPUS" \
  scripts/test_icl_baseline.py \
  --model /data/lz/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --datasets race \
  --num-shots 0 \
  --batch-size 64 \
  --max-input-tokens 8192 \
  --squad-max-new-tokens 32 \
  --race-max-new-tokens 8 \
  --dtype bfloat16 \
  --output-dir outputs/icl_baseline/qwen3-1.7b-0shot \
  "$@"