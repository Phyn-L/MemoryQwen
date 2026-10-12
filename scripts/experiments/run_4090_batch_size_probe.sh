#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DATA_ROOT="${DATA_ROOT:-/data/lz/contexts}"
export MODEL_ROOT="${MODEL_ROOT:-/data/lz/hf_cache/hub}"

CONDA_ENV="${CONDA_ENV:-shine}"
DATASETS="${DATASETS:-c1qa}"
BATCH_SIZES="${BATCH_SIZES:-1,2,4,8}"
ROWS="${ROWS:-8}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/train_baseline.yaml}"
OUTPUT="${OUTPUT:-outputs/benchmarks/4090_batch_size_probe.json}"

mkdir -p "$(dirname "$OUTPUT")"

echo "Running 4090 batch-size probe"
echo "  model config: $MODEL_CONFIG"
echo "  datasets:     $DATASETS"
echo "  batch sizes:  $BATCH_SIZES"
echo "  rows:         $ROWS"
echo "  context:      4096 tokens"
echo "  answer:       128 tokens"
echo "  output:       $OUTPUT"

conda run --no-capture-output -n "$CONDA_ENV" \
  python scripts/benchmark_train_batch_size.py \
  --data-root "$DATA_ROOT" \
  --datasets "$DATASETS" \
  --batch-sizes "$BATCH_SIZES" \
  --rows "$ROWS" \
  --sample-mode extreme \
  --model-config "$MODEL_CONFIG" \
  --machine 4090 \
  --max-context-tokens 4096 \
  --max-conversation-tokens 1024 \
  --max-answer-tokens 128 \
  --output "$OUTPUT"
