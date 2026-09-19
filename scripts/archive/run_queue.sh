#!/usr/bin/env bash
# 串行队列：跑完一个再跑下一个，中途失败**默认就停**（不然会在错误的前提上再烧几个小时）。
#
#   QUEUE=8b,m32,m16 bash scripts/archive/run_queue.sh     # 默认：8B ON -> ctx1024/M32(32:1) -> ctx1024/M16(64:1)
#   QUEUE=m32,m16 bash scripts/archive/run_queue.sh        # 跳过 8B（8B 已经跑完/在别处跑）
#   QUEUE=8b,eval8b,m32,m16 bash scripts/archive/run_queue.sh   # 8B 跑完先做全集评测，再接着训形状
#   DRYRUN=1 QUEUE=8b,m32,m16 bash scripts/archive/run_queue.sh # 只做预检 + 打印每个阶段的命令
#   STOP_ON_FAIL=0 bash scripts/archive/run_queue.sh       # 失败也继续（想一口气把能跑的都跑掉时用）
#
# 阶段：
#   8b       Qwen3-8B + ON 开关（scripts/archive/run_on_8b.sh，默认 batch 4 x 8 卡 = global 32 / 7890 步）
#   eval8b   对 outputs/on_8b 下最新的 last.pt 跑全集 SQuAD v1/v2（4 卡）
#   m32      ctx1024/M=32（32:1）ON 单臂：configs/ablations/memory_length.yaml
#   m16      ctx1024/M=16（64:1）ON 单臂：configs/ablations/memory_length.yaml
#
# 环境变量（都会透传给对应阶段）：
#   MACHINE=h200   NUM_PROCESSES=8   BATCH_SIZE=4（只影响 8b 阶段）  RESUME=1  SMOKE=1
#   STOP_ON_FAIL=1（默认）  DRYRUN=0  EVAL_RANKS=4  QUEUE=8b,m32,m16
#
# 每阶段一个日志：logs/queue_<时间戳>_<阶段>.log，全队汇总：logs/queue_<时间戳>.summary
# 产物目录：outputs/on_8b/（8B）、outputs/shape_m32/、outputs/shape_m16/、outputs/eval_on_8b/（评测）
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# 每台机器自己的设置（gitignored）。train.sh / test.sh / run_on_8b.sh 都 source 它，所以
# PATH / MODEL_ROOT / DATA_ROOT 这类东西写在这里一次就够，不用每条命令都 export。
# 例如 H200 上可以加一行： export PATH=/home/lijie/proj2/.conda/envs/shine/bin:$PATH
if [ -f "$ROOT/scripts/env.local.sh" ]; then
  # shellcheck source=/dev/null
  . "$ROOT/scripts/env.local.sh"
fi

MACHINE="${MACHINE:-h200}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
QUEUE="${QUEUE:-8b,m32,m16}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
DRYRUN="${DRYRUN:-0}"
EVAL_RANKS="${EVAL_RANKS:-4}"
export MACHINE NUM_PROCESSES DRYRUN

for required in utils src configs pyproject.toml; do
  if [ ! -e "$ROOT/$required" ]; then
    echo "run_queue.sh: $ROOT 下没有 '$required' —— 这不是一个完整的 checkout。" >&2
    exit 1
  fi
done

valid_stage() {
  case "$1" in 8b|eval8b|m32|m16) return 0 ;; *) return 1 ;; esac
}
stage_config() {
  case "$1" in
    m32) echo "configs/ablations/memory_length.yaml" ;;
    m16) echo "configs/ablations/memory_length.yaml" ;;
  esac
}
stage_outdir() {
  case "$1" in
    8b) echo "outputs/on_8b" ;;
    m32) echo "outputs/shape_m32" ;;
    m16) echo "outputs/shape_m16" ;;
    eval8b) echo "outputs/eval_on_8b" ;;
  esac
}
stage_desc() {
  case "$1" in
    8b) echo "Qwen3-8B + ON 开关（global batch $(( ${BATCH_SIZE:-4} * NUM_PROCESSES ))）" ;;
    eval8b) echo "8B checkpoint 全集 SQuAD v1/v2（$EVAL_RANKS 卡）" ;;
    m32) echo "ctx1024/M32 = 32:1，ON 单臂" ;;
    m16) echo "ctx1024/M16 = 64:1，ON 单臂" ;;
  esac
}

STAGES=()
IFS=',' read -r -a RAW <<<"$QUEUE"
for stage in "${RAW[@]}"; do
  stage="${stage// /}"
  [ -n "$stage" ] || continue
  if ! valid_stage "$stage"; then
    echo "run_queue.sh: 未知阶段 '$stage'（可选：8b / eval8b / m32 / m16）" >&2
    exit 1
  fi
  STAGES+=("$stage")
done
[ "${#STAGES[@]}" -gt 0 ] || { echo "run_queue.sh: QUEUE 是空的" >&2; exit 1; }

echo "=== 队列：${STAGES[*]} ==="
echo "machine=$MACHINE  NUM_PROCESSES=$NUM_PROCESSES  STOP_ON_FAIL=$STOP_ON_FAIL  DRYRUN=$DRYRUN"
for stage in "${STAGES[@]}"; do
  printf '  %-7s %-46s -> %s\n' "$stage" "$(stage_desc "$stage")" "$(stage_outdir "$stage")"
done

# ---- 预检：先确认每个要用的 config 都在、都能 validate、模型路径都存在 ----
# 这一步的意义是"别等 8B 跑完 5 小时才发现 m32 的 config 打不开"。
for stage in "${STAGES[@]}"; do
  case "$stage" in
    8b|eval8b) continue ;;   # 8B 的配置由 run_on_8b.sh 自己生成并校验；评测脚本自己会校验
  esac
  config=$(stage_config "$stage")
  if [ ! -f "$config" ]; then
    echo "run_queue.sh: 阶段 $stage 的配置不存在：$config" >&2
    exit 1
  fi
done
if printf '%s\n' "${STAGES[@]}" | grep -qE '^(m32|m16)$'; then
  PREFLIGHT=$(printf '%s\n' "${STAGES[@]}" | grep -E '^(m32|m16)$' | while read -r s; do stage_config "$s"; done)
  # 注意：配置清单走 argv，不能走 stdin —— `python - <<PY` 的 stdin 已经被脚本本身占用了。
  python - "$MACHINE" "$PREFLIGHT" <<'PY' || exit 1
import sys
from pathlib import Path
sys.path.insert(0, ".")
from utils.config import TrainConfig

machine, listing = (sys.argv[1] or None), sys.argv[2]
for line in listing.split():
    cfg = TrainConfig.from_file(line, machine=machine)
    cfg.validate()
    path = Path(cfg.model.name_or_path)
    if not path.exists():
        sys.exit(f"run_queue.sh: {line} 的模型路径不存在：{path}")
    print(f"  preflight ok: {line}  M={cfg.memory.memory_length} ctx={cfg.data.max_context_tokens}"
          f" ratio={cfg.data.max_context_tokens // cfg.memory.memory_length}:1 batch={cfg.training.batch_size}"
          f" out={cfg.checkpoint.output_dir}")
PY
fi
if [ "$NUM_PROCESSES" != "8" ] && printf '%s\n' "${STAGES[@]}" | grep -qE '^(m32|m16)$'; then
  echo "  ⚠ m32/m16 的 cadence 是按 batch8 x 8 ranks（global 64 / 3945 步）写死的；" >&2
  echo "    你现在 NUM_PROCESSES=${NUM_PROCESSES}，global batch 变了，严格对齐请用 8 卡。" >&2
fi

STAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="logs/queue_${STAMP}.summary"
mkdir -p logs
: >"$SUMMARY"

stage_command() {
  local stage="$1" ckpt
  case "$stage" in
    8b) echo "bash scripts/archive/run_on_8b.sh" ;;
    eval8b)
      ckpt=$(find outputs/on_8b -maxdepth 2 -name 'last.pt' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | awk '{print $2}')
      [ -n "$ckpt" ] || { echo "run_queue.sh: outputs/on_8b 下没有 last.pt，eval8b 无法执行" >&2; return 1; }
      echo "CHECKPOINT=$ckpt NUM_PROCESSES=$EVAL_RANKS MACHINE=$MACHINE WORK=outputs/eval_on_8b bash scripts/archive/eval_squad_v1v2.sh"
      ;;
    *)
      ckpt=""
      if [ "${RESUME:-0}" != "0" ]; then
        ckpt=$(find "$(stage_outdir "$stage")" -maxdepth 2 -name 'step-*.pt' -printf '%f %p\n' 2>/dev/null \
          | sed 's/^step-\([0-9]*\)\.pt /\1 /' | sort -n | tail -1 | awk '{print $2}')
      fi
      echo "CONFIG=$(stage_config "$stage") NUM_PROCESSES=$NUM_PROCESSES MACHINE=$MACHINE bash scripts/train.sh ${ckpt:+--resume $ckpt}"
      ;;
  esac
}

if [ "$DRYRUN" != "0" ]; then
  echo
  echo "=== DRYRUN：每个阶段将会执行 ==="
  for stage in "${STAGES[@]}"; do
    echo "  [$stage] $(stage_command "$stage" || echo '<不可用>')"
  done
  echo "(DRYRUN=1，未启动。8B 阶段的配置校验由 scripts/archive/run_on_8b.sh 的 DRYRUN 自己做，"
  echo " 想单独看就跑 DRYRUN=1 bash scripts/archive/run_on_8b.sh)"
  exit 0
fi

failed=0
for stage in "${STAGES[@]}"; do
  log="logs/queue_${STAMP}_${stage}.log"
  echo
  echo "=== [$stage] $(stage_desc "$stage")  $(date '+%H:%M:%S') -> $log"
  command=$(stage_command "$stage") || { failed=1; break; }
  echo "    $command"
  # 用 bash -c + tee：日志一份、终端一份，退出码按被 pipe 的命令算（pipefail）。
  set -o pipefail
  MACHINE="$MACHINE" NUM_PROCESSES="$NUM_PROCESSES" \
    BATCH_SIZE="${BATCH_SIZE:-4}" RESUME="${RESUME:-0}" SMOKE="${SMOKE:-0}" \
    bash -c "$command" 2>&1 | tee "$log"
  status=$?
  if [ "$status" -eq 0 ]; then
    printf '%-7s ok   %s\n' "$stage" "$log" >>"$SUMMARY"
    echo "=== [$stage] 完成（exit 0）"
  else
    printf '%-7s FAIL (exit %d)   %s\n' "$stage" "$status" "$log" >>"$SUMMARY"
    # ${status} 的花括号不能省：后面紧跟全角括号，bash 3.2 会把它的首字节当成变量名的一部分。
    echo "=== [$stage] 失败（exit ${status}），日志：$log" >&2
    failed=1
    [ "$STOP_ON_FAIL" != "0" ] && break
  fi
done

echo
echo "=== 队列结束 $(date '+%H:%M:%S') ==="
cat "$SUMMARY"
echo "汇总：$SUMMARY"
if [ -n "$(ls -d outputs/shape_m32/*/ outputs/shape_m16/*/ outputs/on_8b/*/ 2>/dev/null)" ]; then
  echo "checkpoint 目录："
  ls -d outputs/on_8b/*/ outputs/shape_m32/*/ outputs/shape_m16/*/ 2>/dev/null | sed 's/^/  /'
fi
exit "$failed"
