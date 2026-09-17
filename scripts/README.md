# Shell 使用指南

日常运行只使用三个 shell。它们不保存实验超参数：训练、评测与 ICL 变体都由 YAML 声明；shell 只定位仓库、选择可见 GPU，并将参数交给 Python。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

`--machine` 解析 [utils/machines.py](../utils/machines.py) 中的 `4090` 或 `h200` 数据/模型根路径。除非需要临时覆盖，本机 PATH 或 CUDA 设置可放在不提交的 `scripts/env.local.sh`。

| 目标 | 命令 |
|---|---|
| 按训练 YAML 训练 | `bash scripts/train.sh --machine 4090 --config configs/qwen-1.7b/train_baseline.yaml` |
| 从 checkpoint 测试 HotpotQA | `bash scripts/test.sh --machine 4090 --ckpt outputs/<run>/last.pt --datasets hotpotqa --bs 2` |
| 用 Qwen ICL 测试 HotpotQA | `bash scripts/icl_baseline_test.sh --machine h200 --model Qwen3-8B --config configs/icl/icl_hotpotqa_4shot.yaml --bs 4` |

`icl_test.sh` 是第三个入口的简短别名。`--bs` 在 memory test 中表示每进程 context 数，在 ICL 中表示每 GPU QA 数。三个入口默认每张可见 GPU 一个进程；仅在需要少于可见 GPU 的进程时再设置 `NUM_PROCESSES`。

## 参数边界

- `train.sh`：`--config` 是完整训练 YAML；所有训练设置均由 YAML 读取。恢复时增加 `--resume <checkpoint>`。
- `test.sh`：`--ckpt` 必需，默认从 checkpoint 恢复模型结构；`--datasets`、`--bs`、`--split` 是允许的运行时覆盖。测试协议可以由 `configs/evaluation/test_<dataset>.yaml` 给出。
- `icl_baseline_test.sh`：`--model` 使用本地 Qwen family 名称或完整模型路径；`--config` 选择 `configs/icl/icl_<dataset>_<shots>shot.yaml`。模型名会从当前 machine 的缓存中解析，不下载模型、不在多 snapshot 间猜测。

HotpotQA 当前使用 `validation.jsonl`：本地 `test.jsonl` 为空。memory 与 ICL 都只报告答案指标，不报告 supporting-fact 或 joint 指标。

## YAML 命名

- `configs/qwen-<size>/train_baseline.yaml`
- `configs/qwen-<size>/train_reader-{on,off}_ctx<L>_m<M>.yaml`
- `configs/evaluation/test_<dataset>.yaml`
- `configs/icl/icl_<dataset>_<N>shot.yaml`

旧的 A/B、8B、队列、专用 SQuAD 评测和早期 ICL wrapper 已移至 [archive](archive/README.md)。它们用于追溯历史实验，不是日常入口；新的实验变体应写入 YAML，而不是新增专用 shell。
