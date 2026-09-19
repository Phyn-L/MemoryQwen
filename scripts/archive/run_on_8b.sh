#!/usr/bin/env bash
# Qwen3-8B + 六个 reader 开关全开（ON）的单跑脚本。
#
# 为什么这样配：第二轮在 1.7B 上拿到了干净的一对 A/B（同 ctx1024/M64 = 16:1、同 global 64、
# 同 1 epoch、同一份代码），ON 全面领先（全集 v1：AR f1 0.6445 vs 0.5448、TF f1 0.6842 vs 0.5844、
# memory 探针 7.289 vs 7.402）。这一步只把 backbone 换成 Qwen3-8B，其余（开关 / 形状 / 数据 /
# 调度比例 / 评测口径）逐字段沿用那份 ON 臂配置，这样"8B vs 1.7B"的差就只来自 backbone。
#
# 8B 在 H200（每卡 140 GB）上是可行的：ON 开关里的 A1（tied 词表头）让 context-LM 打分头
# **零参数**（`TiedUnembedding` / `VocabularyHead(mode=tied)`），否则光一个 fp32 的
# [151936, 4096] 分类头就是 2.5 GB 参数 + 7.5 GB 优化器状态；8B 主干 bf16 权重 16.4 GB，
# LoRA + memory + decoders 的可训练量约 20M（fp32 + AdamW 约 0.24 GB）。**24 GB 的 4090 放不下
# 这个形状**（AE 那一趟是全 context 的第二趟前向），所以默认按 H200 写。
#
# 用法（在 GPU 节点的 tmux 里跑；`nvidia-smi` 在登录节点不可用）：
#
#   cd /home/lijie/proj2/xmu/lz/MemoryQwen
#   export PATH=/home/lijie/proj2/.conda/envs/shine/bin:$PATH
#
#   DRYRUN=1 bash scripts/archive/run_on_8b.sh          # 只生成配置 + 打印调度，不启动
#   SMOKE=1  bash scripts/archive/run_on_8b.sh          # 256 个 context 的冒烟（十几步，几分钟）
#   bash scripts/archive/run_on_8b.sh                   # 正式跑（默认 batch 4/卡 x 8 卡 = global 32）
#   BATCH_SIZE=8 bash scripts/archive/run_on_8b.sh      # 与第二轮 ON 完全同调度（global 64 / 3945 步）
#   RESUME=1 bash scripts/archive/run_on_8b.sh          # 从 $WORK 下最新的 checkpoint 续跑
#
# 环境变量（都有默认值）：
#   MACHINE=h200        取 utils/machines.py 里的 MODEL_ROOT / DATA_ROOT（也可写进 config）
#   NUM_PROCESSES=8     默认取可见 GPU 数
#   BATCH_SIZE=4        每卡 batch；4 -> global 32 -> 1 epoch 7890 步；8 -> global 64 -> 3945 步
#   CTX=1024  M=64      context 上限 / memory token 数（16:1，与第二轮 ON 相同）
#   AE_POSITIONS=128    AE / context-LM / distill 的采样位置数（显存紧张就降到 64）
#   TRAIN_DATASETS=all  训练集（"all" 或逗号分隔，如 "squad,coqa,drop"）
#   WORK=outputs/on_8b  生成的配置 + checkpoint 的父目录
#   BASE_CONFIG=configs/train_baseline.yaml   逐字段继承的 ON 臂配置
#   MODEL=...           8B 快照（默认 ${MODEL_ROOT}/models--Qwen--Qwen3-8B/snapshots/b968826d...）
#   SMOKE=1 DRYRUN=1 RESUME=1
#
# 产物：
#   $WORK/train.yaml                       这次实际生效的配置（自包含，可以直接复用/改）
#   $WORK/Qwen8B_<时间戳>/step-*.pt,last.pt
#   logs/on_8b_<时间戳>.log
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ROOT

for required in utils src configs pyproject.toml; do
  if [ ! -e "$ROOT/$required" ]; then
    echo "run_on_8b.sh: $ROOT has no '$required' -- this is not a complete checkout." >&2
    exit 1
  fi
done

# 每台机器自己的设置（gitignored，train.sh / test.sh 也 source 它）。PATH 这类东西写在这里一次就够：
#   export PATH=/home/lijie/proj2/.conda/envs/shine/bin:$PATH
# 它在本脚本读下面那些默认值之前生效，所以这里设的 MACHINE / BATCH_SIZE 等也会被采纳。
if [ -f "$ROOT/scripts/env.local.sh" ]; then
  # shellcheck source=/dev/null
  . "$ROOT/scripts/env.local.sh"
fi

MACHINE="${MACHINE:-h200}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CTX="${CTX:-1024}"
M="${M:-64}"
AE_POSITIONS="${AE_POSITIONS:-128}"
TRAIN_DATASETS="${TRAIN_DATASETS:-all}"
WORK="${WORK:-outputs/on_8b}"
BASE_CONFIG="${BASE_CONFIG:-configs/train_baseline.yaml}"
MODEL="${MODEL:-}"
SMOKE="${SMOKE:-0}"
DRYRUN="${DRYRUN:-0}"
RESUME="${RESUME:-0}"
SMOKE_LIMIT="${SMOKE_LIMIT:-256}"
# 冒烟跑写到另一个目录：它的 train_max_samples 与正式跑不同，别让 RESUME=1 捡到它的 checkpoint。
if [ "$SMOKE" != "0" ] && [ "$WORK" = "outputs/on_8b" ]; then
  WORK="outputs/on_8b_smoke"
fi

# `python` 必须是项目环境（这台机器的 PATH 里可能排着别的 conda）：生成配置要它的 yaml，
# 真正启动时 scripts/train.sh 还会再查一遍 torch。
PYTHON_BIN=$(command -v python || true)
if [ -z "$PYTHON_BIN" ] || ! python -c "import yaml" >/dev/null 2>&1; then
  echo "run_on_8b.sh: 'python' -> ${PYTHON_BIN:-<none>} 不能 import yaml。" >&2
  echo "             先 export PATH=/home/lijie/proj2/.conda/envs/shine/bin:\$PATH" >&2
  echo "             （torch 由 scripts/train.sh 自己再查一遍；这里不 import torch，" >&2
  echo "              因为登录节点上它要几十秒，而 DRYRUN 用不到。）" >&2
  exit 1
fi
ACCELERATE_BIN=$(command -v accelerate || true)
if [ -n "$ACCELERATE_BIN" ] && [ "$(dirname "$ACCELERATE_BIN")" != "$(dirname "$PYTHON_BIN")" ]; then
  echo "run_on_8b.sh: python ($PYTHON_BIN) 与 accelerate ($ACCELERATE_BIN) 不在同一个环境。" >&2
  exit 1
fi

if [ ! -f "$ROOT/$BASE_CONFIG" ] && [ ! -f "$BASE_CONFIG" ]; then
  echo "run_on_8b.sh: BASE_CONFIG '$BASE_CONFIG' 不存在。" >&2
  exit 1
fi

# 进程数：显式给就用显式的，否则数可见 GPU；CUDA_VISIBLE_DEVICES 决定"可见"。
# 注意登录节点看不到 GPU（nvidia-smi 返回 0 行），这时按 8 卡估算调度并提示 —— 绝不悄悄退化成 1 卡，
# 否则 cadence 会被按 63k 步推导，正式跑就全错了。真正启动前还有一道"看不到 GPU 就不启动"的闸。
if [ -z "${NUM_PROCESSES:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_PROCESSES=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
  else
    NUM_PROCESSES=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    if [ "${NUM_PROCESSES:-0}" -lt 1 ]; then
      NUM_PROCESSES=8
      echo "run_on_8b.sh: 这台机器看不到 GPU，按 NUM_PROCESSES=8 估算调度。" >&2
      echo "              请在 GPU 节点（tmux）里跑，或用 NUM_PROCESSES=N 显式指定。" >&2
    fi
  fi
fi
case "$NUM_PROCESSES" in ''|*[!0-9]*) NUM_PROCESSES=1 ;; esac
[ "$NUM_PROCESSES" -ge 1 ] || NUM_PROCESSES=1

# 1 epoch 的 context 数（train_datasets=all、filter_long_context=true 后剩下的量）。
# 这是**估算**：训练脚本启动时会打印权威的 `schedule: steps=...`，用那一行核对。
case "$CTX" in
  1024) CONTEXTS="${CONTEXTS:-252465}" ;;
  2048) CONTEXTS="${CONTEXTS:-255143}" ;;
  *)    CONTEXTS="${CONTEXTS:-252465}"
        echo "run_on_8b.sh: CTX=$CTX 不在 1024/2048 里，按 $CONTEXTS 个 context 估算调度；" >&2
        echo "              以训练脚本打印的 steps= 为准（cadence 是比例推导的，偏一点没关系）。" >&2 ;;
esac
CONTEXTS="${CONTEXTS:-252465}"
export BATCH_SIZE CTX M AE_POSITIONS TRAIN_DATASETS WORK BASE_CONFIG MODEL MACHINE \
       SMOKE SMOKE_LIMIT CONTEXTS NUM_PROCESSES

mkdir -p "$WORK" logs
GEN_CONFIG="$WORK/train.yaml"

# 生成配置：从 ON 臂逐字段继承，只改 backbone / 形状 / 显存 / 规模这些字段，并按
# "1 epoch 的步数" 推导 cadence（warmup 5%、TF 20 点、AR 10 点、checkpoint 跟随 AR）。
python - "$ROOT/$BASE_CONFIG" "$GEN_CONFIG" <<'PY'
import math, os, sys
from pathlib import Path
import yaml

base_path, out_path = sys.argv[1], sys.argv[2]
if not Path(base_path).exists():
    base_path = os.environ["BASE_CONFIG"]
cfg = yaml.safe_load(Path(base_path).read_text(encoding="utf-8"))

env = os.environ
batch, ranks, ctx, m = int(env["BATCH_SIZE"]), int(env["NUM_PROCESSES"]), int(env["CTX"]), int(env["M"])
smoke = env["SMOKE"] not in ("", "0", "false")
limit = int(env["SMOKE_LIMIT"]) if smoke else None
contexts = limit if limit else int(env["CONTEXTS"])
steps = max(1, math.ceil(contexts / (batch * ranks)))

def nice(x: float) -> int:
    x = max(1, int(round(x)))
    for candidate in (10, 20, 25, 50, 100, 200, 250, 400, 500, 800, 1000, 1500, 2000, 2500, 4000, 5000):
        if candidate >= x:
            return candidate
    return int(round(x / 1000.0)) * 1000

if smoke:
    warmup, tf, ar = 1, steps, steps
else:
    warmup, tf, ar = nice(steps * 0.05), nice(steps / 20), nice(steps / 10)

datasets = env["TRAIN_DATASETS"].strip()
cfg["data"]["train_datasets"] = datasets if datasets == "all" else [d.strip() for d in datasets.split(",") if d.strip()]
cfg["data"]["train_max_samples"] = limit
cfg["data"]["max_context_tokens"] = ctx
cfg["memory"]["memory_length"] = m
positions = int(env["AE_POSITIONS"])
for key in ("ae_lm_positions", "context_lm_positions", "distill_positions"):
    cfg["memory"][key] = positions
cfg["training"]["batch_size"] = batch
cfg["scheduler"]["warmup_steps"] = warmup
cfg["evaluation"]["teacher_forced_every"] = tf
cfg["evaluation"]["autoregressive_every"] = ar
cfg["checkpoint"]["output_dir"] = env["WORK"]
cfg["logging"]["log_every"] = 25
if env["MODEL"]:
    cfg["model"]["name_or_path"] = env["MODEL"]
else:
    # 保留 ${MODEL_ROOT:-...} 占位符：由 utils/machines.py 的机器表在运行时展开。
    cfg["model"]["name_or_path"] = (
        "${MODEL_ROOT:-/data/lz/hf_cache/hub}"
        "/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    )

Path(out_path).write_text(yaml.safe_dump(cfg, sort_keys=True, allow_unicode=True), encoding="utf-8")

# 用项目自己的加载器校验一遍（顺便把 ${MODEL_ROOT} 按机器表展开、确认权重真的在）。
sys.path.insert(0, env["ROOT"])
from utils.config import TrainConfig  # noqa: E402

machine = env.get("MACHINE") or None
resolved = TrainConfig.from_file(out_path, machine=machine)
resolved.validate()
model_path = Path(resolved.model.name_or_path)
if not model_path.exists():
    sys.exit(
        f"run_on_8b.sh: 8B 权重不存在: {model_path}\n"
        f"              MACHINE={machine!r} 的 MODEL_ROOT 下没有这个快照；"
        f"用 MODEL=/abs/path 显式指定。"
    )

switch = {k: getattr(resolved.memory, k) for k in
          ("head_mode", "init_mode", "slot_attention", "ae_lm_weight", "distill_weight", "readout_length")}
print("=== 本次配置（继承自 " + env["BASE_CONFIG"] + "） ===")
print(f"  backbone      : {model_path}")
print(f"  ON 开关       : {switch}")
print(f"  形状          : ctx={resolved.data.max_context_tokens} M={resolved.memory.memory_length}"
      f" ({resolved.data.max_context_tokens // max(1, resolved.memory.memory_length)}:1)"
      f"  qa_per_context={resolved.data.qa_per_context} max_answer_tokens={resolved.data.max_answer_tokens}")
print(f"  采样位置      : ae/context_lm/distill = {resolved.memory.ae_lm_positions}"
      f"/{resolved.memory.context_lm_positions}/{resolved.memory.distill_positions}")
print(f"  数据          : train_datasets={resolved.data.train_datasets} train_max_samples={resolved.data.train_max_samples}")
print(f"  规模(估算)    : contexts={contexts} global batch={batch}x{ranks}={batch * ranks}"
      f" -> 1 epoch = {steps} 步")
print(f"  cadence       : warmup={warmup} TF={tf} AR={ar} checkpoint={ar} log={resolved.logging.log_every}")
print(f"  评测口径      : val_max_samples={resolved.data.validation_max_samples}"
      f" AR_max_qa={resolved.evaluation.autoregressive_max_qa} max_new_tokens={resolved.evaluation.max_new_tokens}")
print(f"  输出          : {resolved.checkpoint.output_dir}/Qwen8B_<时间戳>/")
PY

LOADER_ARGS=()
if [ "$RESUME" != "0" ]; then
  LATEST=$(find "$WORK" -maxdepth 2 -name 'step-*.pt' -printf '%f %p\n' 2>/dev/null \
    | sed 's/^step-\([0-9]*\)\.pt /\1 /' | sort -n | tail -1 | awk '{print $2}')
  if [ -n "${LATEST:-}" ]; then
    echo "resume: $LATEST"
    LOADER_ARGS=(--resume "$LATEST")
  else
    echo "resume: $WORK 下没有 step-*.pt，从头开始"
  fi
fi

echo
echo "launch: MACHINE=$MACHINE NUM_PROCESSES=$NUM_PROCESSES CONFIG=$GEN_CONFIG bash scripts/train.sh ${LOADER_ARGS[*]:-}"
if [ "$DRYRUN" != "0" ]; then
  echo "(DRYRUN=1，未启动)"
  exit 0
fi

# 最后一道闸：看不到 GPU 就直接停（登录节点上跑下去只会 8B 权重都加载不了）。
VISIBLE_NOW=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
if [ "${VISIBLE_NOW:-0}" -lt 1 ]; then
  echo "run_on_8b.sh: 这台机器看不到 GPU（登录节点？）。请在 GPU 节点的 tmux 里跑。" >&2
  exit 1
fi
if [ "$NUM_PROCESSES" -gt "$VISIBLE_NOW" ]; then
  echo "run_on_8b.sh: NUM_PROCESSES=$NUM_PROCESSES 但只看到 $VISIBLE_NOW 张卡；改成 ${VISIBLE_NOW}。" >&2
  NUM_PROCESSES="$VISIBLE_NOW"
fi

LOG="logs/on_8b_$(date +%Y%m%d_%H%M%S).log"
echo "log   : $LOG"
MACHINE="$MACHINE" NUM_PROCESSES="$NUM_PROCESSES" CONFIG="$GEN_CONFIG" \
  bash scripts/train.sh "${LOADER_ARGS[@]+"${LOADER_ARGS[@]}"}" 2>&1 | tee "$LOG"

CKPT=$(find "$WORK" -maxdepth 2 -name 'last.pt' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | awk '{print $2}')
echo
echo "训练结束。用同一把尺子（全集 SQuAD v1/v2，4 卡）评它："
echo "  CHECKPOINT=${CKPT:-$WORK/Qwen8B_<时间戳>/last.pt} NUM_PROCESSES=4 WORK=outputs/eval_on_8b \\"
echo "    bash scripts/archive/eval_squad_v1v2.sh"
