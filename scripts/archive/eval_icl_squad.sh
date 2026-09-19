#!/usr/bin/env bash
# SQuAD validation 的 ICL 基线：Qwen3-1.7B 与 Qwen3-8B，全集、few-shot、贪心 32 token。
#
# 目的：给 memory 侧（`scripts/archive/eval_squad_v1v2.sh` 的 AR 数字）一个同口径的对照。两边共用
# `src/metrics.answer_line` + 官方 SQuAD 归一化 + "每条参考取 max" 的归约（见 docs/AB_H200.md
# 第 3 节），生成都是贪心 + `max_new_tokens=32`，评分都只取生成结果的第一行。
#
# 子集（SUBSETS，默认三个都跑）：
#   v1    10,570 行 = <SPLIT_DIR>/v1/squad/validation.jsonl  —— memory 侧 v1.1 那一列的同批行
#   v2     5,928 行 = <SPLIT_DIR>/v2/squad/validation.jsonl  —— 只含有答案的行（无答案行两个
#                                                              harness 都会跳过，见 src/icl_baseline.load_jsonl）
#   full  16,498 行 = <DATA_ROOT>/squad/validation.jsonl 里全部有答案的行（= v1 + v2 有答案的行，
#                     其中 5,928 行是 v2 对 v1 问题的重复标注），与训练期 validation 是同一批行
#   v2all 与 v2 逐行相同（无答案行在数据层就被丢掉，见 docs/EVAL_ANOMALIES.md 第 1 节），默认不跑。
#
# 用法：
#   cd <repo>
#   MACHINE=h200 bash scripts/archive/eval_icl_squad.sh                  # 1.7B + 8B，4-shot，v1/v2/full
#   MACHINE=h200 MODELS="8b" SHOTS=0 SUBSETS="v1" bash scripts/archive/eval_icl_squad.sh
#   MACHINE=h200 DRYRUN=1 bash scripts/archive/eval_icl_squad.sh          # 只打印将要执行的命令
#   MACHINE=h200 SAMPLE_CAP=8 MODELS="1.7b" SUBSETS="v1" bash scripts/archive/eval_icl_squad.sh   # 冒烟
#   MACHINE=h200 RESUME=1 MODELS="8b" bash scripts/archive/eval_icl_squad.sh                      # 断点续跑
#
# 常用环境变量：
#   MODELS="1.7b 8b"     要跑的模型（1.7b / 8b）
#   SHOTS="4"            few-shot 例数，可给多个：SHOTS="0 4"（0 复现 9 月 16 日那批 0-shot）
#   SUBSETS="v1 v2 full"
#   MODEL_1P7B / MODEL_8B  显式指定权重目录（默认在 MODEL_ROOT 下按快照目录找）
#   BATCH_SIZE_1P7B=32 / BATCH_SIZE_8B=16   每卡 batch（按行数）
#   SPLIT_DIR=outputs/eval_on    v1/v2 子集文件的来源（eval_squad_v1v2.sh 的 WORK 目录）
#   WORK=outputs/icl_baseline    产物根目录
#   SAMPLE_CAP=               调试用：每个子集只跑前 N 行
#   MAX_NEW_TOKENS=32 MAX_INPUT_TOKENS=8192 NUM_GPUS=<可见卡数> NUM_WORKERS=0
#   NO_CHAT_TEMPLATE=1 RESUME=1 DRYRUN=1
#
# 产物：
#   outputs/icl_baseline/qwen3-<tag>-<shots>shot/<subset>/squad.metrics.json  （+ squad.predictions.jsonl）
#   outputs/icl_baseline/results.md                                            （汇总表）
#   logs/icl_<tag>_<shots>shot_<subset>.log
#
# 注意：`--resume` 的分片文件名里带 rank 数，续跑必须用同样的 NUM_GPUS。
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

# Same completeness check as train.sh/test.sh: a stray copy of scripts/ next to the data tree
# would otherwise start 8 ranks that all die with "No module named 'utils'".
for required in utils src scripts; do
  if [ ! -e "$ROOT/$required" ]; then
    echo "eval_icl_squad.sh: $ROOT has no '$required' -- run this from a complete checkout." >&2
    exit 1
  fi
done

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# The project environment must own `python`: a shell whose PATH puts another conda first gives a
# python without torch, and the failure surfaces much later inside every rank.
PYTHON=${PYTHON:-$(command -v python || true)}
if [ -z "$PYTHON" ]; then
  echo "eval_icl_squad.sh: no 'python' on PATH" >&2
  exit 1
fi
if ! "$PYTHON" -c "import torch" >/dev/null 2>&1; then
  echo "eval_icl_squad.sh: 'python' resolves to $PYTHON, which cannot import torch." >&2
  echo "eval_icl_squad.sh: activate the project environment first (conda activate <env>)." >&2
  exit 1
fi

# Which machine this is -- and therefore where the weights and the data live -- is one field:
# `MACHINE=h200 bash scripts/archive/eval_icl_squad.sh`, resolved against utils/machines.py exactly like
# train.sh / test.sh do. The gitignored scripts/env.local.sh still works and still wins: the table
# is only consulted for what is still unset.
if [ -f "$ROOT/scripts/env.local.sh" ]; then
  # shellcheck source=/dev/null
  . "$ROOT/scripts/env.local.sh"
fi
if [ -z "${MODEL_ROOT:-}" ] || [ -z "${DATA_ROOT:-}" ]; then
  eval "$("$PYTHON" - "${MACHINE:-}" <<'PY'
import os, shlex, sys
from utils.machines import fill_missing, machine_environ, resolve_machine

name = sys.argv[1] or None
resolved, authoritative = resolve_machine(name)
values = machine_environ(resolved)
if not authoritative:
    values = fill_missing(values, os.environ)
print(f"export MACHINE_NAME={shlex.quote(resolved)}")
for key in ("MODEL_ROOT", "DATA_ROOT"):
    if values.get(key):
        print(f"export {key}={shlex.quote(values[key])}")
PY
)"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

MODELS="${MODELS:-1.7b 8b}"
SHOTS="${SHOTS:-4}"
SUBSETS="${SUBSETS:-v1 v2 full}"
BATCH_SIZE_1P7B="${BATCH_SIZE_1P7B:-32}"
BATCH_SIZE_8B="${BATCH_SIZE_8B:-16}"
SPLIT_DIR="${SPLIT_DIR:-outputs/eval_on}"
WORK="${WORK:-outputs/icl_baseline}"
SAMPLE_CAP="${SAMPLE_CAP:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-8192}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MODEL_ROOT="${MODEL_ROOT:-/data/lz/hf_cache/hub}"
DATA_ROOT="${DATA_ROOT:-/data/lz/contexts/aggregated}"
DRYRUN="${DRYRUN:-0}"

DETECTED=$("$PYTHON" -c "import torch; print(torch.cuda.device_count())")
NUM_GPUS="${NUM_GPUS:-$DETECTED}"
case "$NUM_GPUS" in ''|*[!0-9]*) NUM_GPUS=1 ;; esac
if [ "$NUM_GPUS" -lt 1 ]; then
  if [ "$DRYRUN" != "0" ]; then
    # DRYRUN only prints commands, so it is also usable from the login node (no GPU there).
    NUM_GPUS=1
    echo "eval_icl_squad.sh: no GPU visible; DRYRUN still works and assumes NUM_GPUS=1." >&2
  else
    echo "eval_icl_squad.sh: no visible GPU (this looks like the login node)." >&2
    echo "eval_icl_squad.sh: run it on the GPU node, in tmux." >&2
    exit 1
  fi
fi
if [ "$DETECTED" -gt 0 ] && [ "$NUM_GPUS" -gt "$DETECTED" ]; then
  echo "eval_icl_squad.sh: NUM_GPUS=$NUM_GPUS but only $DETECTED GPU(s) visible" \
       "(CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}); running $DETECTED." >&2
  NUM_GPUS="$DETECTED"
fi

resolve_model() {  # $1 = 1.7B | 8B
  local tag="$1" dir="$MODEL_ROOT/models--Qwen--Qwen3-$1/snapshots" found count
  found=$(find "$dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)
  count=$(printf '%s\n' "$found" | grep -c . || true)
  if [ "$count" -eq 0 ]; then
    echo "eval_icl_squad.sh: no Qwen3-$tag snapshot under $dir" >&2
    echo "eval_icl_squad.sh: pass MODEL_${tag//./} explicitly, or fix MODEL_ROOT." >&2
    exit 1
  fi
  if [ "$count" -gt 1 ]; then
    echo "eval_icl_squad.sh: $count Qwen3-$tag snapshots under $dir; using the last one." >&2
    echo "eval_icl_squad.sh: pin it with MODEL_${tag//./} if that is not the intended revision." >&2
  fi
  printf '%s' "$(printf '%s\n' "$found" | tail -1)"
}

validation_file() {  # $1 = subset
  case "$1" in
    full) printf '%s' "$DATA_ROOT/squad/validation.jsonl" ;;
    v1|v2|v2all) printf '%s' "$SPLIT_DIR/$1/squad/validation.jsonl" ;;
    *) echo "eval_icl_squad.sh: unknown subset '$1' (use v1|v2|v2all|full)" >&2; exit 1 ;;
  esac
}

# Fail before launching 16 ranks if a split file is missing: they come from the memory-side
# evaluation script, which is also what guarantees both harnesses score the same rows.
for subset in $SUBSETS; do
  file=$(validation_file "$subset")
  if [ ! -f "$file" ]; then
    echo "eval_icl_squad.sh: subset '$subset' needs '$file', which does not exist." >&2
    echo "eval_icl_squad.sh: generate the splits once with any checkpoint, e.g." >&2
    echo "    CHECKPOINT=<ckpt> SAMPLE_CAP=1 bash scripts/archive/eval_squad_v1v2.sh" >&2
    echo "  (SAMPLE_CAP only caps the decoded rows; the split files are always written in full)" >&2
    echo "eval_icl_squad.sh: or point SPLIT_DIR at an existing WORK directory." >&2
    exit 1
  fi
done

mkdir -p "$WORK" logs

echo "machine    : ${MACHINE_NAME:-${MACHINE:-<from hostname>}}  MODEL_ROOT=$MODEL_ROOT  DATA_ROOT=$DATA_ROOT"
echo "models     : $MODELS      shots: $SHOTS      subsets: $SUBSETS"
echo "gpus       : NUM_GPUS=$NUM_GPUS (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all $DETECTED>})"
echo "decoding   : greedy, max_new_tokens=$MAX_NEW_TOKENS, max_input_tokens=$MAX_INPUT_TOKENS," \
     "chat_template=$([ -n "${NO_CHAT_TEMPLATE:-}" ] && echo off || echo on)"
echo "rows       : ${SAMPLE_CAP:+first $SAMPLE_CAP of each subset}${SAMPLE_CAP:-all rows of each subset}"
echo "work       : $WORK      split_dir=$SPLIT_DIR"

# Row counts come from the loader the evaluation itself uses, so a wrong file (or v2all, which is
# v2 in disguise) is visible here instead of after twenty minutes of GPU time.
"$PYTHON" - "$SUBSETS" "$DATA_ROOT" "$SPLIT_DIR" <<'PY'
import sys
from src.icl_baseline import load_jsonl

subsets, data_root, split_dir = sys.argv[1].split(), sys.argv[2], sys.argv[3]
for subset in subsets:
    path = (f"{data_root}/squad/validation.jsonl" if subset == "full"
            else f"{split_dir}/{subset}/squad/validation.jsonl")
    print(f"[rows]  {subset:<5} {len(load_jsonl(path, 'squad')):6d} answerable rows   {path}")
PY

status=0
for shots in $SHOTS; do
  for tag in $MODELS; do
    case "$tag" in
      1.7b) upper="1.7B"; batch="$BATCH_SIZE_1P7B"; override="${MODEL_1P7B:-}" ;;
      8b)   upper="8B";   batch="$BATCH_SIZE_8B";   override="${MODEL_8B:-}" ;;
      *) echo "eval_icl_squad.sh: unknown model tag '$tag' (use 1.7b|8b)" >&2; exit 1 ;;
    esac
    model="${override:-$(resolve_model "$upper")}"
    for subset in $SUBSETS; do
      file=$(validation_file "$subset")
      out="$WORK/qwen3-$tag-${shots}shot/$subset"
      log="logs/icl_${tag}_${shots}shot_${subset}.log"
      extra=()
      [ -n "$SAMPLE_CAP" ] && extra+=(--max-samples "$SAMPLE_CAP")
      [ -n "${NO_CHAT_TEMPLATE:-}" ] && extra+=(--no-chat-template)
      [ -n "${RESUME:-}" ] && extra+=(--resume)
      echo
      echo "=== qwen3-$tag / ${shots}-shot / $subset -> $out"
      if [ "$DRYRUN" != "0" ]; then
        echo "  $PYTHON -m torch.distributed.run --standalone --nproc-per-node=$NUM_GPUS \\"
        echo "    scripts/evaluation/test_icl_baseline.py --datasets squad --model $model \\"
        echo "    --squad-validation-file $file --squad-train-file $DATA_ROOT/squad/train.jsonl \\"
        echo "    --num-shots $shots --batch-size $batch --num-workers $NUM_WORKERS \\"
        echo "    --max-input-tokens $MAX_INPUT_TOKENS --squad-max-new-tokens $MAX_NEW_TOKENS \\"
        echo "    --dtype bfloat16 --seed 42 --output-dir $out ${extra[*]}"
        continue
      fi
      if "$PYTHON" -m torch.distributed.run \
          --standalone --nproc-per-node="$NUM_GPUS" \
          scripts/evaluation/test_icl_baseline.py \
          --datasets squad \
          --model "$model" \
          --squad-validation-file "$file" \
          --squad-train-file "$DATA_ROOT/squad/train.jsonl" \
          --num-shots "$shots" \
          --batch-size "$batch" \
          --num-workers "$NUM_WORKERS" \
          --max-input-tokens "$MAX_INPUT_TOKENS" \
          --squad-max-new-tokens "$MAX_NEW_TOKENS" \
          --dtype bfloat16 \
          --seed 42 \
          --output-dir "$out" \
          "${extra[@]}" 2>&1 | tee "$log"; then
        :
      else
        status=1
        echo "eval_icl_squad.sh: run failed (qwen3-$tag ${shots}-shot $subset); see $log" >&2
      fi
    done
  done
done

if [ "$DRYRUN" != "0" ]; then
  echo
  echo "eval_icl_squad.sh: DRYRUN -- nothing was executed."
  exit 0
fi

# One table over every (model, shots, subset) that has a metrics file. run_config.json sits next to
# the metrics and belongs to that single run (one output directory per model/shots/subset), so the
# table can show the shot count and per-GPU batch that actually produced each row.
"$PYTHON" - "$WORK" <<'PY'
import json, sys
from pathlib import Path

work = Path(sys.argv[1])
order = {"v1": 0, "v2": 1, "full": 2, "v2all": 3}
runs: dict[str, list] = {}
for metrics_path in sorted(work.glob("qwen3-*/*/squad.metrics.json")):
    subset, run_dir = metrics_path.parent.name, metrics_path.parent.parent
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config_path = metrics_path.parent / "run_config.json"
    shots = batch = None
    if config_path.exists():
        args = json.loads(config_path.read_text(encoding="utf-8")).get("arguments", {})
        shots, batch = args.get("num_shots"), args.get("batch_size")
    runs.setdefault(run_dir.name, []).append((order.get(subset, 9), subset, metrics, shots, batch))

if not runs:
    print(f"no squad.metrics.json under {work}; nothing to summarise")
    raise SystemExit(0)

lines = [
    "# SQuAD validation ICL 基线（贪心，max_new_tokens=32，取生成结果第一行）", "",
    "| run | subset | rows | shots | batch/GPU | EM | F1 | ROUGE-L |",
    "| --- | --- | --- | --- | --- | --- | --- | --- |",
]
for run in sorted(runs):
    for _, subset, metrics, shots, batch in sorted(runs[run]):
        lines.append(
            f"| {run} | {subset} | {metrics.get('count')} | {shots} | {batch} | "
            f"{metrics.get('em', 0):.4f} | {metrics.get('f1', 0):.4f} | {metrics.get('rouge_l', 0):.4f} |"
        )
table = "\n".join(lines) + "\n"
(work / "results.md").write_text(table, encoding="utf-8")
print()
print(table)
print(f"written: {work / 'results.md'}")
PY

echo
echo "eval_icl_squad.sh: done (exit=$status). Logs in logs/icl_*.log, table in $WORK/results.md"
exit "$status"
