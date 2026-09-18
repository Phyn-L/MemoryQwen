# Shell 使用指南

日常运行只使用三个 shell。它们不保存实验超参数：训练、评测与 ICL 变体都由 YAML 声明；shell 只定位仓库、选择可见 GPU，并将参数交给 Python。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

`--machine` 解析 [utils/machines.py](../utils/machines.py) 中的 `4090` 或 `h200` 数据/模型根路径。除非需要临时覆盖，本机 PATH 或 CUDA 设置可放在不提交的 `scripts/env.local.sh`。

| 目标 | 命令 |
|---|---|
| 按训练 YAML 训练 | `bash scripts/train.sh --machine 4090 --config configs/4090/qwen-1.7b/baseline/train_baseline.yaml` |
| 从 checkpoint 测试 HotpotQA | `bash scripts/test.sh --machine 4090 --ckpt outputs/<run>/last.pt --datasets hotpotqa --bs 2` |
| 用 Qwen ICL 测试 HotpotQA | `bash scripts/icl_baseline_test.sh --machine h200 --model Qwen3-8B --config configs/4090/icl/icl_hotpotqa_4shot.yaml --bs 4` |

`icl_test.sh` 是第三个入口的简短别名。`--bs` 在 memory test 中表示每进程 context 数，在 ICL 中表示每 GPU QA 数。三个入口默认每张可见 GPU 一个进程；仅在需要少于可见 GPU 的进程时再设置 `NUM_PROCESSES`。

## 参数边界

- `train.sh`：`--config` 是完整训练 YAML；所有训练设置均由 YAML 读取。恢复时增加 `--resume <checkpoint>`。
- `test.sh`：`--ckpt` 必需，默认从 checkpoint 恢复模型结构；`--datasets`、`--bs`、`--split` 是允许的运行时覆盖。测试协议可以由 `configs/4090/evaluation/test_<dataset>.yaml` 给出。
- `icl_baseline_test.sh`：`--model` 使用本地 Qwen family 名称或完整模型路径；`--config` 选择 `configs/4090/icl/icl_<dataset>_<shots>shot.yaml`。模型名会从当前 machine 的缓存中解析，不下载模型、不在多 snapshot 间猜测。

HotpotQA 当前使用 `validation.jsonl`：本地 `test.jsonl` 为空。memory 与 ICL 都只报告答案指标，不报告 supporting-fact 或 joint 指标。

## YAML 命名

- `configs/qwen-<size>/train_baseline.yaml`
- `configs/qwen-<size>/train_reader-{on,off}_ctx<L>_m<M>.yaml`
- `configs/4090/evaluation/test_<dataset>.yaml`
- `configs/4090/icl/icl_<dataset>_<N>shot.yaml`

旧的 A/B、8B、队列、专用 SQuAD 评测和早期 ICL wrapper 已移至 [archive](archive/README.md)。它们用于追溯历史实验，不是日常入口；新的实验变体应写入 YAML，而不是新增专用 shell。

## HotpotQA 顺序对比

在 4×4090 上按默认设置连续运行 memory checkpoint、Qwen3-1.7B ICL 和 Qwen3-8B ICL：

```bash
bash scripts/test_hotpotqa_suite.sh --machine 4090
```

默认使用 `outputs/Qwen1.7B_20260917_213648.pt`、4-shot、4 个分布式进程；memory 每卡 context batch 为 1，ICL 每卡 QA batch 为 4。可用 `--ckpt`、`--memory-bs`、`--icl-bs`、`--shots 0|4` 调整。每个结果 JSON 都包含 `em`、`f1`、`rouge_l`。

## MS MARCO v1/v2 顺序对比

```bash
bash scripts/test_ms_marco_suite.sh --machine 4090
```

按顺序测试 checkpoint、Qwen3-1.7B ICL 和 Qwen3-8B ICL 的 MS MARCO `test` split，默认 4 个进程、memory 每卡 context batch 1、ICL 每卡 QA batch 4、0-shot。当前聚合文件含 v1.1（9,650 条有答案 QA）和 v2.1（101,092 条但答案字段为空）；现有答案指标对 v2.1 不具备有效参考意义。

## 全数据集评测

```bash
bash scripts/test_all_suite.sh --machine h200
# 仅打印计划、检查数据，不加载模型
bash scripts/test_all_suite.sh --machine 4090 --dry-run
```

顺序评测 checkpoint、Qwen3-1.7B、Qwen3-8B；每个任务使用所有可评分 QA，不设采样上限。SQuAD v1/v2 按本地聚合文件合并评测；MS MARCO v1.1 分别评测 test 与 validation；v2.1 使用 validation。其他数据集优先非空 test，否则 validation。无参考答案的 split 标记 unscorable，不自动替换成 validation。SQuAD 无答案题仍遵循当前 loader 被排除，因此不是官方完整 v2 无答案协议。memory 关闭长上下文删除，但仍按 checkpoint 的长度截断；ICL 默认 0-shot、8192 输入上限。两者输入预算不同，报告不能解释为相同上下文预算对照。

默认 memory batch 1、ICL batch 2（每卡），自动使用可见 GPU；可设置 CUDA_VISIBLE_DEVICES 和 NUM_PROCESSES。输出在带时间戳的 outputs/all_suite 子目录，包括逐任务日志及 manifest.json。训练 checkpoint、模型缓存、数据不经 Git 同步，运行前需在机器上存在。
