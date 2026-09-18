# 配置索引

规范：训练配置使用 `train_<用途>[_ctx<L>_m<M>].yaml`，位于对应模型规模目录；测试协议使用 `evaluation/test_<数据集>.yaml`；ICL 使用 `icl/icl_<数据集>_<N>shot.yaml`。机器不是实验名的一部分，通过 `--machine` 选择。配置自包含，不引入隐式继承。

## 规范训练文件与兼容旧名

| 旧路径 | 规范文件（同目录） |
|---|---|
| `qwen-1.7b/ab_h200_off.yaml` | [train_reader-off_ctx1024_m64.yaml](qwen-1.7b/train_reader-off_ctx1024_m64.yaml) |
| `qwen-1.7b/ab_h200_on.yaml` | [train_reader-on_ctx1024_m64.yaml](qwen-1.7b/train_reader-on_ctx1024_m64.yaml) |
| `qwen-1.7b/ab_h200_on_ctx2048.yaml` | [train_reader-on_ctx2048_m64.yaml](qwen-1.7b/train_reader-on_ctx2048_m64.yaml) |
| `qwen-1.7b/ab_h200_on_m16.yaml` | [train_reader-on_ctx1024_m16.yaml](qwen-1.7b/train_reader-on_ctx1024_m16.yaml) |
| `qwen-1.7b/ab_h200_on_m32.yaml` | [train_reader-on_ctx1024_m32.yaml](qwen-1.7b/train_reader-on_ctx1024_m32.yaml) |
| `qwen-1.7b/on_4090_all.yaml` | [train_reader-on_ctx1024_m8.yaml](qwen-1.7b/train_reader-on_ctx1024_m8.yaml) |
| `qwen-1.7b/train.yaml` | [train_baseline.yaml](qwen-1.7b/train_baseline.yaml) |
| `qwen-4b/train.yaml` | [train_baseline.yaml](qwen-4b/train_baseline.yaml) |
| `qwen-8b/train.yaml` | [train_baseline.yaml](qwen-8b/train_baseline.yaml) |

旧名是指向规范 YAML 的符号链接，不维护第二份副本。三个 `train_baseline.yaml` 保持原参数；ON/OFF 和 M/ctx 变体也保留原参数（含旧 machine 字段，可由 CLI 覆盖）。新增 [8B ON](qwen-8b/train_reader-on_ctx1024_m64.yaml) 固定原 8B 启动脚本在 8 GPU、batch 4 时的默认调度；GPU 数改变时需审阅调度，不在 shell 自动重写。

## 测试与 ICL

- [test_hotpotqa.yaml](evaluation/test_hotpotqa.yaml)
- [test_race.yaml](evaluation/test_race.yaml)
- [test_squad.yaml](evaluation/test_squad.yaml)
- [icl_hotpotqa_0shot.yaml](icl/icl_hotpotqa_0shot.yaml)
- [icl_hotpotqa_4shot.yaml](icl/icl_hotpotqa_4shot.yaml)
- [icl_race_0shot.yaml](icl/icl_race_0shot.yaml)
- [icl_race_4shot.yaml](icl/icl_race_4shot.yaml)
- [icl_squad_0shot.yaml](icl/icl_squad_0shot.yaml)
- [icl_squad_4shot.yaml](icl/icl_squad_4shot.yaml)

测试 YAML 只覆盖评测协议，模型结构来自 checkpoint。ICL YAML 提供模型名、数据与推理配置，CLI 只需覆盖机器、模型、数据集或 bs。详细优先级见[运行指南](../docs/guides/RUNNING.md)。HotpotQA 当前仅报告答案指标，不包含 supporting-fact/joint 分数。

## 原训练配置参数快照

L/M 是长度上限与 memory slot 数之比，不是实际压缩率。以下旧链接仍解析到唯一规范 YAML。

| 配置 | 用途 / 模型 | L / M | batch | reader 变体 | 机器 | 实验记录 |
|---|---|---|---|---|---|---|
| [qwen-1.7b/ab_h200_off.yaml](qwen-1.7b/ab_h200_off.yaml) | qwen-1.7b / reader OFF | 1024 / 64 | 8 | head=linear; AE=0.0; KL=0.0; R=0 | 建议 h200，未固定 machine | [A/B](../docs/experiments/AB_H200.md) |
| [qwen-1.7b/ab_h200_on.yaml](qwen-1.7b/ab_h200_on.yaml) | qwen-1.7b / reader ON | 1024 / 64 | 8 | head=tied; AE=1.0; KL=0.3; R=8 | 建议 h200，未固定 machine | [A/B](../docs/experiments/AB_H200.md) |
| [qwen-1.7b/ab_h200_on_ctx2048.yaml](qwen-1.7b/ab_h200_on_ctx2048.yaml) | qwen-1.7b / reader ON | 2048 / 64 | 4 | head=tied; AE=1.0; KL=0.3; R=8 | h200 | [第三轮](../docs/experiments/ROUND3_PLAN.md) |
| [qwen-1.7b/ab_h200_on_m16.yaml](qwen-1.7b/ab_h200_on_m16.yaml) | qwen-1.7b / reader ON | 1024 / 16 | 8 | head=tied; AE=1.0; KL=0.3; R=8 | h200 | [第三轮](../docs/experiments/ROUND3_PLAN.md) |
| [qwen-1.7b/ab_h200_on_m32.yaml](qwen-1.7b/ab_h200_on_m32.yaml) | qwen-1.7b / reader ON | 1024 / 32 | 8 | head=tied; AE=1.0; KL=0.3; R=8 | h200 | [第三轮](../docs/experiments/ROUND3_PLAN.md) |
| [qwen-1.7b/on_4090_all.yaml](qwen-1.7b/on_4090_all.yaml) | qwen-1.7b / reader ON | 1024 / 8 | 1 | head=tied; AE=1.0; KL=0.3; R=8 | 建议 4090，未固定 machine | [实验索引](../docs/experiments/README.md) |
| [qwen-1.7b/train.yaml](qwen-1.7b/train.yaml) | qwen-1.7b / 通用基线 | 2048 / 8 | 1 | head=linear; AE=0.0; KL=0.0; R=0 | 环境/自动解析 | [实验索引](../docs/experiments/README.md) |
| [qwen-4b/train.yaml](qwen-4b/train.yaml) | qwen-4b / 通用基线 | 2048 / 8 | 1 | head=linear; AE=0.0; KL=0.0; R=0 | 环境/自动解析 | [实验索引](../docs/experiments/README.md) |
| [qwen-8b/train.yaml](qwen-8b/train.yaml) | qwen-8b / 通用基线 | 2048 / 8 | 1 | head=linear; AE=0.0; KL=0.0; R=0 | 环境/自动解析 | [实验索引](../docs/experiments/README.md) |

## 两台机器的配置版本

同目录内 `4090_<name>.yaml` 保存本地配置，`h200_<name>.yaml` 保存 H200 配置；两者都纳入仓库。无前缀的原文件保留以兼容现有命令。选择配置时同时显式传入匹配的 `--machine`，例如：

```bash
bash scripts/train.sh --machine h200 --config configs/qwen-1.7b/h200_train_baseline.yaml
```

代码以本地工作区为准，通过 `bash scripts/sync_h200.sh` 同步到 H200。修改 H200 参数时先修改本地仓库中的 `h200_` 配置，再同步。
