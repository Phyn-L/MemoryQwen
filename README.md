# Qwen MetaLoRA Context Memory

使用同一 Qwen backbone 将 context 编码为可复用的 memory KV，再根据 question 生成答案。训练更新 LoRA、memory slots 和所启用的辅助模块；默认使用内置 StaticLoRA，可配置 PEFT。辅助目标包括 memory-only context-LM，以及可选的 AE-LM 和蒸馏。

## 三个运行命令

先激活项目环境，执行 `pip install -e '.[train]'`，通过 `CUDA_VISIBLE_DEVICES` 选择 GPU。所有实验变体写在 YAML 中。

```bash
bash scripts/train.sh --machine 4090 --config configs/qwen-1.7b/train_baseline.yaml
bash scripts/test.sh --machine 4090 --ckpt outputs/<run>/last.pt --datasets hotpotqa --bs 2
bash scripts/icl_baseline_test.sh --machine h200 --model Qwen3-8B --config configs/icl/icl_hotpotqa_4shot.yaml --bs 4
```

`<run>` 替换为实际运行目录。test 默认恢复 checkpoint 配置；HotpotQA 预设使用带答案的 validation split。完整参数、机器路径与 batch 语义见[运行指南](docs/guides/RUNNING.md)。

## 从这里查找

| 想了解什么 | 入口 |
|---|---|
| 环境、训练、恢复、多机路径 | [运行指南](docs/guides/RUNNING.md) |
| 指标、SQuAD 子集和数据过滤 | [评测协议](docs/guides/EVALUATION.md) |
| memory、KV cache、梯度、dtype、辅助目标 | [方法与实现](docs/guides/METHOD.md) |
| reader 开关及约束 | [reader 选项](docs/guides/READER_OPTIONS.md) |
| 每份 YAML 的用途 | [配置索引](configs/README.md) |
| A/B、压缩比、8B 的实验记录 | [实验索引](docs/experiments/README.md) |
| 历史修复与可视化 | [文档导航](docs/README.md) |

## 代码位置

- `src/`：模型、数据、损失、评测和 ICL 基线。
- `utils/`：配置、机器设置、checkpoint、优化器与调度。
- `scripts/`：现有训练、评测、实验及数据入口；`test.py` 是模型评测入口。
- `tests/`：回归测试。
- `configs/`：按模型规模组织的完整 YAML。
- `outputs/`、`logs/`、`wandb/`：本地运行产物，保持现有路径并由 Git 忽略。

文档中的历史实验数字不代表当前重新测量；当前使用说明、历史证据和待验证计划分别索引。目录整理范围见[整理方案](docs/REPOSITORY_LAYOUT_PLAN.md)。
