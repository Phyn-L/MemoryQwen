# Qwen MetaLoRA Context Memory

使用同一 Qwen backbone 将 context 编码为可复用的 memory KV，再根据 question 生成答案。训练更新 LoRA、memory slots 和所启用的辅助模块；默认使用内置 StaticLoRA，可配置 PEFT。辅助目标包括 memory-only context-LM，以及可选的 AE-LM 和蒸馏。

## 三个运行命令

先激活项目环境，执行 `pip install -e '.[train]'`，通过 `CUDA_VISIBLE_DEVICES` 选择 GPU。所有实验变体写在 YAML 中。

```bash
bash scripts/train.sh --machine 4090 --config configs/4090/qwen-1.7b/baseline/train_baseline.yaml
python -m utils.launcher test --machine 4090 --ckpt outputs/<run>/last.pt --datasets hotpotqa --bs 2
python -m utils.launcher icl --machine h200 --model Qwen3-8B --config configs/4090/icl/icl_hotpotqa_4shot.yaml --bs 4
```

`<run>` 替换为实际运行目录。test 默认恢复 checkpoint 配置；HotpotQA 预设使用带答案的 validation split。完整参数、机器路径与 batch 语义见[运行指南](docs/guides/RUNNING.md)。

## 从 4090 同步到 H200 / A800

**4090 是代码和 YAML 配置的开发基准，H200/A800 只负责执行计算任务。** 在 4090 的项目根目录提交修改后，通过增量 Git bundle 同步，无需计算服务器访问 GitHub，也无需先 push：

```bash
# 先提交本次需要同步的代码和 YAML（将路径替换为实际修改文件）
git add <文件路径...>
git commit -m "说明本次修改"

# 选择一个目标，或同步两台服务器
bash scripts/sync_remote.sh h200
bash scripts/sync_remote.sh a800
bash scripts/sync_remote.sh all
```

| 目标 | 默认 SSH 主机 | 默认仓库目录 |
|---|---|---|
| h200 | `h200` | `/home/lijie/proj2/xmu/lz/MemoryQwen` |
| a800 | `A800` | `/mnt/beegfs-storage-a8002/seanlz/MemoryQwen` |

脚本查询远端 HEAD，确认它是本地 HEAD 的祖先，生成增量 bundle，经 SSH 传输后执行快进更新；两端提交相同时不传输 bundle。同步包含所有已提交文件的新增、修改和删除，包括 YAML，不包含未提交修改、未跟踪数据、模型权重或运行产物。

- 本地存在未提交修改时停止，请先整理并提交。
- 远端存在未提交修改、历史分叉或本地未知的提交时停止，不自动 stash 或 reset。需要保留的远端代码/配置改动应先整理回 4090，再提交同步。
- `all` 会尝试两台服务器，任一失败则最终返回非零退出码。
- 运行任务时通过 `--machine h200` 或 `--machine a800` 选择机器路径，避免在计算服务器修改公共 YAML。同步前确认正在运行的任务不会受到代码更新影响；脚本不停止或重启任务。

需要 SSH 非交互认证和远端 Git、Bash、`flock`。可通过 `H200_SSH_HOST`、`H200_REPO`、`A800_SSH_HOST`、`A800_REPO` 覆盖目标，仓库路径使用绝对路径。日常统一使用 [sync_remote.sh](scripts/sync_remote.sh)；旧 `sync_h200.sh` / `sync_a800.sh` 仍采用 GitHub 拉取。更多脚本说明见 [Scripts 使用指南](scripts/README.md)。

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

当前推荐 checkpoint 编排入口为 `bash scripts/evaluation/test_all_suite.sh`。旧 shell wrappers 和 `scripts/archive/` 已删除；文中的历史配置路径需按 configs/README.md 选择当前配置。
