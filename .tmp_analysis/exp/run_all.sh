#!/usr/bin/env bash
# Train 4 ablation arms in parallel (one per 4090), then evaluate each.
set -u
EXP=/data/lz/MemoryQwen/.tmp_analysis/exp
PY=/home/lz/miniconda3/envs/shine/bin/python
mkdir -p "$EXP/logs" "$EXP/results"

ARMS=(armA_control armB_fixed armC_fixed_mem64 armD_fixed_norecon)
GPUS=(0 1 2 3)

echo "=== phase 1: training 4 arms in parallel  $(date -Is)"
pids=()
for i in "${!ARMS[@]}"; do
  arm="${ARMS[$i]}"; gpu="${GPUS[$i]}"
  mkdir -p "$EXP/run/$arm"
  (
    for _ in $(seq 1 180); do
      used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
      [ "${used:-99999}" -lt 2000 ] && break
      sleep 10
    done
    echo "gpu $gpu free (used=${used}MiB) for $arm at $(date -Is)"
    cd "$EXP/run/$arm" || exit 1
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$EXP" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    timeout 5400 "$PY" -u "$EXP/scripts/train.py" --config "$EXP/configs/$arm.yaml" \
      > "$EXP/logs/$arm.train.log" 2>&1
    echo "$arm train exit=$?" >> "$EXP/logs/$arm.status"
  ) &
  pids+=($!)
  sleep 5
done
for p in "${pids[@]}"; do wait "$p"; done
echo "=== phase 1 done $(date -Is)"
for arm in "${ARMS[@]}"; do
  tail -2 "$EXP/logs/$arm.train.log" 2>/dev/null | tr '\n' ' ' | cut -c1-200; echo "  <- $arm"
done

echo "=== phase 2: evaluating 4 arms in parallel  $(date -Is)"
pids=()
for i in "${!ARMS[@]}"; do
  arm="${ARMS[$i]}"; gpu="${GPUS[$i]}"
  ckpt=$(ls -t "$EXP/run/$arm"/outputs/*/last.pt 2>/dev/null | head -1)
  if [ -z "$ckpt" ]; then echo "NO CHECKPOINT for $arm"; continue; fi
  echo "$arm -> $ckpt"
  (
    for _ in $(seq 1 180); do
      used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
      [ "${used:-99999}" -lt 2000 ] && break
      sleep 10
    done
    cd "$EXP/run/$arm" || exit 1
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$EXP" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    timeout 3600 "$PY" -u "$EXP/eval_arm.py" "$EXP/configs/$arm.yaml" "$ckpt" 300 \
      > "$EXP/logs/$arm.eval.log" 2>&1
  ) &
  pids+=($!)
  sleep 5
done
for p in "${pids[@]}"; do wait "$p"; done
echo "=== phase 2 done $(date -Is)"
grep -h -E '^(RESULT|TF first-token)' "$EXP"/logs/*.eval.log 2>/dev/null
echo "=== ALL DONE"
