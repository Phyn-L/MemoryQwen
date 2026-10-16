# MemoryQwen Agent 上手指南

本文件是唯一的项目说明文档，按当前代码与 YAML 维护；根目录 `AGENTS.md` 单独保留为工作指令。更新日期：2026-10-16。不要新增重复说明文档，也不要把历史改造过程作为当前操作指南。

## 环境与常用入口

- 工作目录：`/data/lz/MemoryQwen`；环境：`conda activate shine`。
- 主训练配置：`configs/train_baseline.yaml`，继承 `configs/harness.yaml` 和 `configs/models/qwen3-1.7b.yaml`。
- 训练：`bash scripts/train.sh --machine 4090 --config configs/train_baseline.yaml`。
- 恢复训练：在训练命令后追加 `--resume <checkpoint>`；仅加载模型权重使用 `--init-from <checkpoint>`，两者互斥。
- 原始语料构建：`python scripts/data/build_1qa.py`。
- 过滤缓存构建：`python scripts/data/build_cache.py --config configs/train_baseline.yaml`；仅补长度文件可追加 `--lengths-only`。
- 单 checkpoint 评测：`python scripts/test.py --checkpoint <path> --split validation`；默认恢复 checkpoint 内的配置。
- 评测套件：`bash scripts/evaluation/test_all_suite.sh`；ICL 对照入口为 `scripts/evaluation/test_icl_suite.py`。
- 测试：`PYTHONPATH=. python -m pytest -q`；静态检查：`python -m ruff check .` 和 `git diff --check`。

## 代码导航


| 路径                                                           | 职责                                                           |
| -------------------------------------------------------------- | -------------------------------------------------------------- |
| `configs/`                                                     | 共享 harness、模型定义、实验预设和消融                         |
| `src/model.py`                                                 | Qwen memory 模型、context 编码、QA/Recon、生成与 checkpointing |
| `src/resampler.py`                                             | Context memory 压缩与 question readout                         |
| `src/data.py`、`src/cache_build.py`                            | 数据加载、单一过滤实现、缓存指纹与长度 sidecar                 |
| `src/eval_data.py`                                             | 评测数据集名称与划分路径的唯一解析入口                         |
| `src/pipeline.py`、`src/evaluator.py`                          | loader、评测抽样、多卡评测与指标汇总                           |
| `src/losses.py`                                                | QA、Recon、蒸馏等训练目标                                      |
| `utils/config.py`、`utils/machines.py`                         | YAML 继承、环境变量展开、配置校验、机器路径                    |
| `utils/checkpoint.py`                                          | checkpoint 保存、加载与 best 更新                              |
| `scripts/train.py`、`scripts/test.py`                          | 训练与单 checkpoint 评测入口                                   |
| `scripts/data/`、`scripts/evaluation/`、`scripts/experiments/` | 数据构建、评测编排、实验运行脚本                               |
| `tests/`                                                       | 以 CPU 单元/集成测试为主                                       |

## 配置与数据链路

配置通过 `_base_` 深度合并：`harness.yaml` 放共享训练、数据、目标权重和评测设置，`models/` 放权重、dtype、LoRA 设置，实验预设的 `memory:` 放架构。目标权重在 `loss:`。读取配置使用 `TrainConfig.from_file`，不要直接用 `yaml.safe_load` 代替配置解析。`--machine` 通过 `utils/machines.py` 提供模型与数据路径。

数据位于 `<data.root>/1qa/`：


| 路径                                          | 内容                                                          |
| --------------------------------------------- | ------------------------------------------------------------- |
| `meta/meta_<split>.json`                      | 原始合并划分；内容为 JSONL，一行一个 context 及全部`qa_pairs` |
| `meta/<split>/<slug>.json`                    | 按数据集拆分的原始数据，供评测套件使用                        |
| `meta/build.manifest.json`                    | 构建计数与实际来源划分                                        |
| `cache/train.jsonl`、`cache/validation.jsonl` | 按训练配置过滤后的数据                                        |
| `cache/build.json`                            | 过滤缓存指纹                                                  |
| `cache/<split>.jsonl.lengths.json`            | context token 长度 sidecar                                    |
| `cache/loader/`                               | loader 派生缓存；可由`data.preprocess_cache_dir` 覆盖         |

`build_1qa.py` 不做训练长度过滤和条数截断，只丢弃结构上无效的 context/QA；MS MARCO 只使用 `is_selected=1` 的 passage。context 不借用其他行文本。数据缺失划分或该划分无可用行时按 `test → validation → train` 回退，并记录实际来源。

训练启动会补建缺失或过期的过滤缓存；只有指纹匹配才跳过 loader 再过滤。构建器和 loader 共用 `filter_rows_batch`，不要维护两套规则。sortish 优先复用长度 sidecar；当前长度测量保持串行。`train_max_samples`、`validation_max_samples` 是 loader 上限，不是缓存构建上限。

## 模型、训练与评测

长 context 经 Qwen 编码并由 context resampler 压缩成固定 memory，reader 从 memory 前缀回答问题或重建 context。Qwen 基础权重冻结，配置的手写 LoRA、resampler 和 `recon_task_embedding` 可训练。QA 使用冻结 Qwen token embedding 和 chat template，**没有 QA task embedding**。默认 backbone 为 BF16，可训练参数为 FP32；修改时注意 dtype 转换与梯度流。

`_transformer_body` 仅调用不带 LM head 的解码器主体；缓存该主体时不要注册重复 submodule。生成采用贪心解码。

有效 batch 为 `training.batch_size × gradient_accumulation_steps × world_size`。训练步、调度、评测间隔均按优化器步计。读取数据、训练、QA/Recon 评测使用 Rich 进度显示；训练条包含学习率，启动日志打印模型参数摘要和 checkpoint 间隔。

评测按 `training.seed` 对每个数据集固定抽取最多 `evaluation.max_eval_contexts_per_dataset` 个 context，每个 context 最多取 `data.task.max_qa_sample_per_context` 条 QA（解析后为 `cfg.data.max_qa_sample_per_context`）。QA 自回归与 Recon 复用入选 context；Recon 开关位于 `evaluation`。各 rank 分配入选数据、共同执行评测与指标汇总，不能只让主 rank 调用评测器。

每次评测前先保存步数 checkpoint 并刷新 `last.pt`；评测后按 QA F1 决定是否更新 `best.pt`。修改多卡流程时留意集合通信顺序，避免 rank 提前进入训练同步。

## Context gradient checkpointing

是否开启由 `model.context_gradient_checkpointing` 控制；间隔可手动设置：

```yaml
training:
  gradient_checkpoint_every_n_layers: 2
```

配置类默认 `1`，当前 harness 显式设置 `2`。训练采用 Transformers 原生 `gradient_checkpointing_enable(use_reentrant=False, preserve_rng_state=True)`，仅在有梯度的 context 编码时开启，随后关闭并移除 embedding 梯度 hook。

当前 `every_n_layers=N` 是**稀疏单层开关间隔**：只有 `index % N == 0` 的层使用 checkpoint；不是连续 N 层分段。N=1 表示逐层 checkpoint。PyTorch 连续分段只留在基准脚本中，不进入训练路径。

2026-10-16 实测记录：RTX 4090、Qwen3-1.7B（28 层）、BF16 backbone / FP32 trainable，batch=2、context=1536、memory=64、question/answer 各 32 token、无 padding、联合 QA+Recon loss。峰值为 `torch.cuda.max_memory_allocated`；耗时为 5 次前后向中位数，不含 optimizer、数据和分布式同步。结果只适用于该工作负载。


| 实现                             | N | 峰值 allocated | 秒/前后向 |
| -------------------------------- | -: | -------------: | --------: |
| Transformers native sparse layer | 1 |      12.00 GiB |     0.960 |
| Transformers native sparse layer | 2 |      17.97 GiB |     0.900 |
| Transformers native sparse layer | 4 |      20.93 GiB |     0.873 |
| Transformers native sparse layer | 6 |      21.78 GiB |     0.864 |
| PyTorch contiguous segments      | 1 |      11.99 GiB |     0.962 |
| PyTorch contiguous segments      | 2 |      11.83 GiB |     0.965 |
| PyTorch contiguous segments      | 4 |      11.74 GiB |     0.965 |
| PyTorch contiguous segments      | 6 |      11.71 GiB |     0.968 |

该实验中，稀疏单层 N 越大显存越高；连续分段相对原生 N=1 最多省约 2.4%，没有速度收益，因此训练保留原生实现。

```bash
PYTHONPATH=. python scripts/benchmark_context_checkpointing.py \
  --config configs/train_baseline.yaml --device cuda:0 \
  --batch-size 2 --sequence-length 1536 --intervals 1 2 4 6 \
  --warmup 1 --iterations 5 \
  --output outputs/benchmarks/context_checkpointing_joint_20261016.jsonl
```

已有 CPU 测试对比无 checkpoint 路径的 loss 和所有可训练参数梯度，覆盖多个间隔、padding、LoRA、Dropout 和联合目标。BF16 CUDA 路径报告误差，不要求逐位一致。CPU 测试通过不代表完整多卡训练或显存测量通过；修改模型路径后按需补 GPU 验证。

本次整理验证（2026-10-16，`shine`）：297 tests passed，ruff 与 `git diff --check` 通过；未重新运行 GPU 基准或完整训练。
