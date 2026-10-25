# MemoryQwen Agent 上手指南

本文件是唯一的项目说明文档，按当前代码与 YAML 维护；根目录 `AGENTS.md` 单独保留为工作指令。更新日期：2026-10-22。不要新增重复说明文档，也不要把历史改造过程作为当前操作指南。

## 环境与常用入口

- 工作目录：`/data/lz/MemoryQwen`；环境：`conda activate shine`。
- 主训练配置：`configs/train_baseline.yaml`，继承 `configs/harness.yaml` 和 `configs/models/qwen3-1.7b.yaml`。
- 训练：`bash scripts/train.sh --machine 4090 --config configs/train_baseline.yaml`。
- 恢复训练：在训练命令后追加 `--resume <checkpoint>`；仅加载模型权重使用 `--init-from <checkpoint>`，两者互斥。
- 原始语料构建：`python scripts/data/build_1qa.py`。
- 过滤缓存构建：`python scripts/data/build_cache.py --config configs/train_baseline.yaml`；仅补长度文件可追加 `--lengths-only`。
- 单 checkpoint 评测：`python scripts/test.py --checkpoint <path> --split validation`；默认恢复 checkpoint 内的配置。
- 评测套件：`bash scripts/evaluation/test_all_suite.sh`。
- 测试：`PYTHONPATH=. python -m pytest -q`；静态检查：`python -m ruff check .` 和 `git diff --check`。

训练入口由 `utils.launcher` 使用 `torch.distributed.run` 启动多进程，训练进程通过 Hugging Face Accelerate 管理 DDP/NCCL、梯度累积和 loader 准备。单卡运行时仍使用同一入口，但不会创建多进程组。

## 代码导航


| 路径                                                           | 职责                                                           |
| -------------------------------------------------------------- | -------------------------------------------------------------- |
| `configs/`                                                     | 共享 harness、模型定义和训练配置预设                           |
| `src/model.py`                                                 | Qwen memory 模型、context 编码、QA/Recon、生成与 checkpointing |
| `src/resampler.py`                                             | Context memory 压缩与 question readout                         |
| `src/data.py`、`src/cache_build.py`                            | 数据加载、单一过滤实现、缓存指纹与长度 sidecar                 |
| `src/evaluation/eval_data.py`                                             | 评测数据集名称与划分路径的唯一解析入口                         |
| `src/data_pipeline.py`、`src/evaluation/`                     | 数据装配、评测抽样、多卡评测与指标汇总                         |
| `src/losses.py`                                                | QA、Recon 训练目标与评估统计                                   |
| `utils/config.py`、`utils/machines.py`                         | YAML 继承、环境变量展开、配置校验、机器路径                    |
| `utils/checkpoint.py`                                          | checkpoint 保存、加载与 best 更新                              |
| `scripts/train.py`、`scripts/test.py`                          | 训练与单 checkpoint 评测入口                                   |
| `scripts/data/`、`scripts/evaluation/`、`scripts/experiments/` | 数据构建、评测编排、实验运行脚本                               |
| `tests/`                                                       | 以 CPU 单元/集成测试为主                                       |

## 配置与数据链路

配置通过 `_base_` 深度合并：`harness.yaml` 放共享训练、数据、目标权重和评测设置，`models/` 放权重、dtype、LoRA 设置，训练配置预设的 `memory:` 放架构。消融配置已删除。目标权重在 `loss:`。读取配置使用 `TrainConfig.from_file`，不要直接用 `yaml.safe_load` 代替配置解析。`--machine` 通过 `utils/machines.py` 提供模型与数据路径。

DataLoader 的运行参数按用途分开配置：`data.loader.train_num_workers`、`train_prefetch_factor`、`train_persistent_workers`、`train_pin_memory` 控制训练；`eval_num_workers`、`eval_prefetch_factor`、`eval_persistent_workers`、`eval_pin_memory` 同时控制普通评估、QA 评估和 Recon 评估。训练将 pinned batch 以 `non_blocking=True` 复制到 GPU。

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

Conversation 数据有独立的 `memory.conversation` 重压缩配置；不再配置 `max_conversation_tokens`。`data.max_question_tokens` 仍用于普通 QA 问题长度过滤，同时当前 `src/data.py` 还会对满足条件的短单轮 prompt 调用 `_trim_prompt_parts`，因此不要把它描述为已全局删除。`data.filter_qa_pair_answers` 默认关闭；开启后，来自 `qa_pairs` 的数据行只要有任一 QA 的问题或答案超出配置 token 长度、或为空，就会整行过滤。关闭时这类 QA-pair 行不会因问题/答案长度检查被过滤，后续 collate 仍按长度配置截断。

训练启动会补建缺失或过期的过滤缓存；只有指纹匹配才跳过 loader 再过滤。构建器和 loader 共用 `filter_rows_batch`，不要维护两套规则。sortish 优先复用长度 sidecar；当前长度测量保持串行。`train_max_samples`、`validation_max_samples` 是 loader 上限，不是缓存构建上限。

训练/评测 QA 共用 `src/qa_prompt.py::format_question`。RACE 的 question 文本为 `question\nOptions:\n<option 1>\n<option 2>...`，回答目标是完整选项文本；该格式不追加 `Answer:`。

## 模型、训练与评测

长 context 单独经 Qwen 编码并由 context resampler 压缩成固定长度的 memory embedding；memory resampler 的输入不包含 user prefix。context encoder 可以融合多个 backbone 层；多层融合前统一经过共享的 final norm，避免不同层的表示尺度直接混合。当前只保留单一 memory 输出，不再支持 layerwise memory 输出、逐层 projection、layer gates 或 decoder 注入 hook。

Qwen 基础权重冻结，配置的手写 LoRA、context/question resampler 和 `recon_task_embedding` 可训练。QA 使用冻结 Qwen token embedding 和 chat template，没有 QA task embedding。默认 backbone 为 BF16，可训练参数为 FP32；修改时注意 dtype 转换与梯度流。

训练、QA 推理与 conversation 的完整单轮布局为 `[memory] <|im_start|>user\n [question] [readout] <|im_end|>\n<|im_start|>assistant\n [answer]`。memory 是纯 causal prefix；user header 在 memory 之后，question resampler 不读取 user header。`question_resampler_info_source: hidden` 时，context 与纯 question token 在同一个 padded batch 编码调用中各占独立行，question 行可缓存并与 context resampler 生成的 memory hidden 一起送入 question resampler。context、question 和 memory hidden 均由 LoRA-on 编码器生成；decoder memory KV 则由 LoRA-off 原始 LLM 单独生成，推理与 conversation 复用这份 KV。Conversation 初始化只缓存 memory KV/hidden；每轮 readout 只使用该轮 user body 和 memory，不读取此前 turn 的原始 hidden。Chat template 实际在 `<|im_end|>` 与下一条 `<|im_start|>` 之间输出换行符。`question_resampler_info_source: embedding` 时直接使用 `[memory embedding, question embedding]`，仍不读取 user header。`learned_prefix` 只使用可学习 prefix，`off` 不创建 question resampler；`readout_keep_question_tokens` 控制回答序列是否保留原始 question token。Recon 布局为 `[memory] <|im_start|>user\n [task] <|im_end|>\n<|im_start|>assistant\n [context]`，context 只作为 assistant 侧的监督目标。

`_transformer_body` 仅调用不带 LM head 的解码器主体。生成 prefill 使用 `logits_to_keep=1`，避免物化不需要的全量词表 logits；生成采用贪心解码。

多轮 conversation API 由 `memory.conversation` 控制，默认关闭。`start_conversation` 对 context/resampler 运行一次，并只用 `[memory]` 初始化 decoder cache；`append_conversation_turn` 追加本轮 user header/body、当前 readout、`<|im_end|>\n<|im_start|>assistant\n` 与回答。达到 `recompress_threshold` 或窗口预算不足时，只使用旧 memory 与已完成历史 token 重压缩，不重新读取原始 context；重压缩后重建 cache。每轮完成后按原生历史模板重放本轮 user/assistant 对，移除 generation-only 空 think 块与旧 readout，并重算受其影响的本轮 KV；此前历史与 memory KV 保留。当前 API 只支持 batch size 1。

有效 batch 为 `training.batch_size × gradient_accumulation_steps × world_size`。训练步、调度、评测间隔均按优化器步计。读取数据、训练、QA/Recon 评测使用 Rich 进度显示；训练条包含学习率，启动日志打印模型参数摘要和 checkpoint 间隔。

评测按 `training.seed` 对每个数据集固定抽取最多 `evaluation.max_eval_contexts_per_dataset` 个 context，每个 context 最多取 `data.task.max_qa_sample_per_context` 条 QA（解析后为 `cfg.data.max_qa_sample_per_context`）。QA 自回归与 Recon 复用入选 context；Recon 开关位于 `evaluation`。各 rank 分配入选数据、共同执行评测与指标汇总，不能只让主 rank 调用评测器。

每次评测前先保存步数 checkpoint 并刷新 `last.pt`；评测后按当前阶段指标决定是否更新 `best.pt`。训练进度条按 optimizer step 更新，micro-batch 只在 GPU 上累积 Tensor；分组梯度范数只在 `logging.log_every` 的 step 计算。评估结束后不主动调用 `torch.cuda.empty_cache()`。修改多卡流程时留意集合通信顺序，避免 rank 提前进入训练同步。

训练任务由 `data.task.train_stage` 选择：`only_qa`、`only_recon` 只训练对应任务；`two_stage` 先训练 Recon，再训练 QA，每阶段各运行 `training.epochs` 个完整 epoch，阶段顺序由 epoch 决定。阶段 checkpoint 分别位于 `<run>/recon/` 和 `<run>/qa/`，各自维护 `best.pt` 与 `last.pt`；Recon best 按 validation reconstruction token accuracy 选择，QA best 按 validation F1 选择。从任一阶段 checkpoint 恢复时，`step` 决定全局 epoch，因此从 Recon 阶段末尾恢复会继续进入 QA 阶段。

`mix` 在每个优化 step 按 schedule 为每个 context 独立抽样 QA 与 Recon：`recon_schedule: [start, end]` 和 `qa_schedule: [start, end]` 分别是两种任务的 Bernoulli 概率，允许同一 context 同时执行两种任务。先独立采样；若某 context 两种任务都未抽中，则按条件概率补抽其中一种，保证每个 context 至少有一个任务。令 `p_q`、`p_r` 为当步概率，兜底抽 QA 的概率为 `p_q(1-p_r) / (p_q(1-p_r) + p_r(1-p_q))`；若分母为 0（两者都为 0），QA/Recon 各以 0.5 概率兜底。`recon_schedule_curve: cosine` 控制两个 schedule 在 start 与 end 之间的余弦插值；start 等于 end 时概率恒定。配置概率是兜底前的独立抽样概率，兜底会改变最终任务频率。

训练和单 checkpoint 评估共同使用 `utils/reporting.py::format_evaluation_report`。项目只支持当前 canonical 数据布局和配置字段；旧路径、旧字段别名和旧 checkpoint 迁移分支不再保留。

## Context gradient checkpointing

是否开启由 `model.context_gradient_checkpointing` 控制；间隔可手动设置：

```yaml
training:
  gradient_checkpoint_every_n_layers: 2
```

配置类默认 `1`，当前 harness 显式设置 `2`。训练采用 Transformers 原生 `gradient_checkpointing_enable(use_reentrant=False, preserve_rng_state=True)`，仅在有梯度的 context 编码时开启，随后关闭并移除 embedding 梯度 hook。

当前 `every_n_layers=N` 是**稀疏单层开关间隔**：只有 `index % N == 0` 的层使用 checkpoint；不是连续 N 层分段。N=1 表示逐层 checkpoint。PyTorch 连续分段只留在基准脚本中，不进入训练路径。

已有 CPU 测试对比无 checkpoint 路径的 loss 和所有可训练参数梯度，覆盖多个间隔、padding、LoRA、Dropout 和联合目标。BF16 CUDA 路径允许数值误差；CPU 测试通过不代表完整多卡训练或显存测量通过。修改模型路径后按需补 GPU 验证。
