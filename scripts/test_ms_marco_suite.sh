#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
[ -f "$ROOT/scripts/env.local.sh" ] && source "$ROOT/scripts/env.local.sh"
MACHINE=4090; CKPT="outputs/Qwen1.7B_20260917_213648.pt"; MEMORY_BS=1; ICL_BS=4; SHOTS=0; OUT_ROOT=outputs/ms_marco_suite
while (($#)); do case "$1" in
 --machine) MACHINE=$2; shift 2;; --ckpt) CKPT=$2; shift 2;; --memory-bs) MEMORY_BS=$2; shift 2;; --icl-bs) ICL_BS=$2; shift 2;; --shots) SHOTS=$2; shift 2;; --output-root) OUT_ROOT=$2; shift 2;;
 -h|--help) echo "Usage: $0 [--machine 4090|h200] [--ckpt PATH] [--memory-bs N] [--icl-bs N] [--shots 0|4] [--output-root PATH]"; exit 0;; *) echo "unknown argument: $1" >&2; exit 2;; esac; done
case "$SHOTS" in 0|4) ;; *) echo '--shots must be 0 or 4' >&2; exit 2;; esac
mkdir -p "$OUT_ROOT"; export NUM_PROCESSES="${NUM_PROCESSES:-4}"
for VERSION in ms_marco_v1_1 ms_marco_v2_1; do
  SPLITS=(validation); [ "$VERSION" = ms_marco_v1_1 ] && SPLITS=(test validation)
  for SPLIT in "${SPLITS[@]}"; do
  SUB="$OUT_ROOT/$VERSION/$SPLIT"; mkdir -p "$SUB"
  python -m utils.launcher test --machine "$MACHINE" --ckpt "$CKPT" --datasets ms_marco --source-version "$VERSION" --split "$SPLIT" --bs "$MEMORY_BS" --qa-batch-size 4
  CONFIG="configs/4090/icl/icl_squad_${SHOTS}shot.yaml"
  for MODEL in Qwen3-1.7B Qwen3-8B; do LABEL=$(echo "$MODEL" | tr "[:upper:]" "[:lower:]" | tr -d "-"); python -m utils.launcher icl --machine "$MACHINE" --model "$MODEL" --config "$CONFIG" --datasets ms_marco --source-version "$VERSION" --split "$SPLIT" --bs "$ICL_BS" --output-dir "$SUB/$LABEL-${SHOTS}shot"; done
  done
done
exit 0
python -m utils.launcher test --machine "$MACHINE" --ckpt "$CKPT" --datasets ms_marco --split test --bs "$MEMORY_BS" --qa-batch-size 4
CONFIG="configs/4090/icl/icl_squad_${SHOTS}shot.yaml"
for MODEL in Qwen3-1.7B Qwen3-8B; do LABEL=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -d '-'); python -m utils.launcher icl --machine "$MACHINE" --model "$MODEL" --config "$CONFIG" --datasets ms_marco --split test --bs "$ICL_BS" --output-dir "$OUT_ROOT/$LABEL-${SHOTS}shot"; done
echo "Reports contain count, em, f1, rouge_l. v1.1 has 9,650 QA; v2.1 has 101,092 rows but currently empty answers."
