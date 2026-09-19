#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [ -f "$ROOT/scripts/env.local.sh" ]; then source "$ROOT/scripts/env.local.sh"; fi
MACHINE=4090
CKPT="outputs/Qwen1.7B_20260917_213648.pt"
MEMORY_BS=1
ICL_BS=4
SHOTS=0
OUT_ROOT="outputs/hotpotqa_suite"
while (($#)); do
  case "$1" in
    --machine) MACHINE=$2; shift 2;;
    --ckpt) CKPT=$2; shift 2;;
    --memory-bs) MEMORY_BS=$2; shift 2;;
    --icl-bs) ICL_BS=$2; shift 2;;
    --shots) SHOTS=$2; shift 2;;
    --output-root) OUT_ROOT=$2; shift 2;;
    -h|--help) echo "Usage: bash scripts/evaluation/test_hotpotqa_suite.sh [--machine 4090|h200] [--ckpt PATH] [--memory-bs N] [--icl-bs N] [--shots 0|4] [--output-root PATH]"; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
case "$SHOTS" in 0) ;; *) echo "--shots must be 0 (zero-shot only)" >&2; exit 2;; esac
mkdir -p "$OUT_ROOT"
export NUM_PROCESSES="${NUM_PROCESSES:-4}"
echo "[1/3] memory checkpoint: $CKPT"
python -m utils.launcher test --machine "$MACHINE" --ckpt "$CKPT" --datasets hotpotqa --split validation --bs "$MEMORY_BS" --qa-batch-size 4
CONFIG="configs/icl_zeroshot.yaml"
for MODEL in Qwen3-1.7B Qwen3-8B; do
  LABEL=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -d '-')
  echo "[2/3] ICL $MODEL (${SHOTS}-shot)"
  python -m utils.launcher icl --machine "$MACHINE" --model "$MODEL" --config "$CONFIG" --datasets hotpotqa --split validation --bs "$ICL_BS" --output-dir "$OUT_ROOT/$LABEL-${SHOTS}shot"
done
cat <<MSG
Completed HotpotQA validation suite. Metrics are written as JSON under $OUT_ROOT and the checkpoint directory.
Each report contains count, em, f1, and rouge_l.
MSG
