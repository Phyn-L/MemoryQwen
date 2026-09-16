#!/usr/bin/env bash
set -u
EXP=/data/lz/MemoryQwen/.tmp_analysis/exp2
PY=/home/lz/miniconda3/envs/shine/bin/python
ARMS=(armE_fp32_M8 armF_bf16_M8 armH_fp32_norecon)
GPUS=(0 1 3)
pids=()
for i in "${!ARMS[@]}"; do
  arm="${ARMS[$i]}"; gpu="${GPUS[$i]}"
  ckpt=$(ls -t "$EXP/run/$arm"/outputs/*/last.pt 2>/dev/null | head -1)
  echo "$arm -> $ckpt"
  (
    cd "$EXP/run/$arm" || exit 1
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=/data/lz/MemoryQwen \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    timeout 3600 "$PY" -u "$EXP/eval_arm.py" "$EXP/configs/$arm.yaml" "$ckpt" 300 \
      > "$EXP/logs/$arm.eval.log" 2>&1
  ) &
  pids+=($!)
  sleep 5
done
for p in "${pids[@]}"; do wait "$p"; done
grep -h -E '^(RESULT|TF first)' "$EXP"/logs/*.eval.log
