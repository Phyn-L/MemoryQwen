# 当前配置

配置直接采用新字段，不提供旧 YAML 字段迁移。机器通过 `--machine 4090/h200/a800` 选择；同一实验不再复制机器版本。

配置分三层，实验配置只写自己的差异：

- `harness.yaml`：共享 harness——数据过滤与 DataLoader、`loss`（目标权重/蒸馏）、`optimizer`、`scheduler`、`training`（含 `gradient_accumulation_steps`）、`evaluation`、`logging`、`checkpoint`。它**不能单独运行**（没有 memory 结构、没有模型）。
- `models/qwen3-1.7b.yaml`、`models/qwen3-4b.yaml`、`models/qwen3-8b.yaml`：模型层，只放 `model:`（权重路径、dtype、LoRA）。换模型只改这一处。
- `train_baseline.yaml` 等实验文件：`_base_: [harness.yaml, models/<模型>.yaml]` + `memory:` 架构 + 本实验的差异。`h200_train_baseline.yaml` 继承 `train_baseline.yaml` 再覆盖机器相关尺寸；`train_baseline_qwen3_4b/8b.yaml` 只换模型层。

`train_baseline.yaml`：Qwen3-1.7B、模型窗口 40960、memory 64、reader on、LoRA on。writer 默认使用最终层，也可用 `memory.context_layer_indices` 选择多个 block 输出并学习 softmax 混合。
- `ablations/`：每个变量一个配置。共享的 harness 设置（评估节奏、batch、warmup、验证子集、readout 开关）在 `ablations/_ablation_common.yaml`，每个消融文件只写唯一不同的科学变量，因此 `tests/test_ab_configs.py` 的“一个消融只改一个因子”断言成立。
- `test.yaml`：一个统一的 checkpoint 评估配置，默认 SQuAD validation。
- `icl_zeroshot.yaml`：一个 zero-shot ICL 配置，可通过 CLI 切换数据集和模型。

## 数据

`data.files.root` 指向 `scripts/data/` 写出的数据树（默认 `contexts`，可用 `DATA_ROOT` 或 `--data-root` 覆盖）。1QA 划分解析为 `<root>/1qa/1qa-{train,validation,test}.jsonl`，每条划分都可以用 `data.files.<split>_conversation_file` 单独覆盖。构建顺序：

数据分两步：**原始 meta** 和**按配置过滤的 cache**。

```bash
# 1) 原始数据（不依赖任何配置；从本地 HF cache 重建）
python scripts/data/build_1qa.py                  # 全部数据集 × train/validation/test
python scripts/data/build_1qa.py --datasets squad race --splits validation
python scripts/data/build_1qa.py --dry-run        # 只统计不写盘

# 2) 过滤后的训练/验证集（过滤规则来自训练 YAML）
python scripts/data/build_cache.py --config configs/train_baseline.yaml
python scripts/data/build_cache.py --config configs/train_baseline.yaml --splits validation
python scripts/data/build_cache.py --config configs/train_baseline.yaml --data-root /mnt/x/contexts
python scripts/data/build_cache.py --config configs/train_baseline.yaml --dry-run
python scripts/data/build_cache.py --config configs/train_baseline.yaml --num-workers 16
python scripts/data/build_cache.py --config configs/train_baseline.yaml --no-progress
```

两个脚本都用进程池：`build_1qa.py` 按 (数据集, 划分) 并行（父进程按数据集顺序写，输出与串行逐字节一致），`build_cache.py` 把 tokenize 过滤分块发给 worker（分块同时受行数与字节数限制；结果按顺序回收，输出同样与串行逐字节一致）。默认 worker = `min(8, CPU 数)`，`--num-workers 1` 退回单进程。

实测（真实 `meta_train.json`、20k 行、真实配置）：过滤 581 row/s（串行）→ 1315（2）→ 2608（8）→ 2833（16），即 worker 数在 8~16 后基本饱和。

`data.loader.num_workers` 只在 **loader 还需要自己过滤时**（指纹不匹配、或 `--data-file` 指向原始文件）才影响速度，此时也建议 8 左右。另一个逐 context 的 prepass（`Preparing sortish lengths`，为分桶算长度）**故意保持单进程**：它是一次批量 tokenizer 调用，tokenizer 内部已并行，实测 14224 context/s（串行）好于 4433（2 worker）/ 9985（8 worker），且结果会缓存。

产物：

| 路径 | 内容 |
|---|---|
| `<root>/1qa/meta/meta_<split>.json` | 原始划分：一行一个 context 及其全部 `qa_pairs`，不过滤长度/条数，只丢弃空 context、空 question、无答案的 QA；MS MARCO 的 context 只由 `is_selected=1` 的 passage 拼接 |
| `<root>/1qa/meta/<split>/<slug>.json` | 同一批行的按数据集拆分（评测套件读这里） |
| `<root>/1qa/meta/build.manifest.json` | 每数据集/划分的 context 与 QA 数 |
| `<root>/1qa/cache/train.jsonl`、`cache/validation.jsonl` | 按 YAML 过滤后的训练/验证集（**训练与 eval 实际读这两个文件**） |
| `<root>/1qa/cache/build.json` | 每个 cache 文件的过滤指纹（配置/规则/meta 文件），用于判断是否需要重建 |
| `<root>/1qa/cache/<split>.jsonl.lengths.json` | 过滤时**顺便记下**的每个 context token 长度（过滤本来就要算 `max_context_tokens`），让 sortish 分桶不再自己扫一遍 |

`build_cache.py` 复用加载器同一个过滤函数（`src/data.py::filter_rows_batch`），所以"缓存里的行"就是"loader 会读的行"；行的 `qa_pairs` 全部保留，按 epoch 抽 QA 仍由 collate 做。`scripts/train.py` 启动时会自己检查 cache 是否存在/过期，缺失或指纹不符就自动构建（多卡用文件锁，只构建一次）；因此手动构建只是为了提前跑或共用同一个数据根。

`data.limits` 里的 `max_context_tokens`（同时是模型侧的训练 context 上限）、`max_question_tokens`、`max_answer_tokens`、`max_conversation_tokens`、`max_input_tokens` 是训练与验证共用的一处全局配置，不再有 `validation_max_context_tokens`。旧的 `/data/lz/contexts/aggregated` 语料已弃用。

## 运行名称

通过 `logging.aim_run_name: my_experiment` 显式指定运行名称，同时用于 Aim
和 checkpoint 子目录：`<checkpoint.output_dir>/my_experiment/`。
省略该字段、设为 `null` 或空字符串时，自动使用模型名和时间，例如
`Qwen1.7B_20260922_153000`。显式名称应使用单个目录名；相同名称会复用同一路径。
`--resume` 始终保留 checkpoint 所在目录。

## 损失字段

所有目标相关的设置都在 `loss:` 段，不再混在 `memory:` 里；旧字段（如 `memory.qa_weight`）会直接报错，不会静默退回默认值。

| 字段 | 意义 | 关闭方式 |
|---|---|---|
| `loss.qa_weight` | QA 答案交叉熵权重 | 0 |
| `loss.recon_weight` | Recon 的 context token 交叉熵权重 | 0 |
| `loss.distill_weight` | prefix-KV student 的分布蒸馏；可独立于 CE 开启 | 0 |
| `loss.distill_positions` / `distill_temperature` / `distill_topk` / `distill_entropy_weight` | 蒸馏位置预算、softmax 温度、teacher top-k、按 teacher 熵加权 | 位置 0 = 全部 |
| `loss.max_logits_rows` | 每次词表投影的行数上限（显存与核数量的权衡，不改变数值） | — |
| `data.train_tasks` / `data.recon_ratio` | `qa`、`recon` 或混合训练；混合时 Recon 样本比例 | `mix` / 0.5 |
| `data.num_workers` / `pin_memory` / `persistent_workers` | 训练与评估 DataLoader 的并行加载和 pinned memory 设置 | `8` / `true` / `true` |
| `training.gradient_accumulation_steps` | 每个优化器步累积的 micro-batch 数（有效 batch = `batch_size × accum × 卡数`），loss 按该值求平均 | 1 |
| `evaluation.max_eval_sample_per_dataset` | 按 `training.seed` 每个数据集独立随机抽取至多 N 条 QA，无合计上限；Recon 复用这些 QA 对应的去重 context；贪心解码 | 2560 |

训练任务通过 `data.train_tasks` 和动态 Recon 调度控制：

```yaml
data:
  train_tasks: mix   # mix、qa、recon
  recon_ratio: 0.5   # constant 调度的 Recon 比例
  recon_schedule: constant
  recon_schedule_curve: linear
  recon_ratio_start: 0.5
  recon_ratio_end: 0.5
  recon_ratio_min: 0.1
```

`mix` 为每个 context 同时生成一条 QA 行和一条 Recon 行，比例调度只改变损失权重。Recon reader 使用 `recon_task_embedding` 和 memory prefix 预测 context；QA 使用 memory 和 conversation 消息，并应用模型原生 chat template。
task embedding 只有一个可训练的 Recon 连续向量，不新增 tokenizer special token，也不改变词表。需要先做 Recon
再做 QA 时，把第一阶段 checkpoint 传给 `scripts/train.py --init-from`，第二阶段设
`train_tasks: qa`。

baseline 默认使用 `mix` 训练，其中 QA 和 Recon 各占一半；蒸馏默认关闭。蒸馏开启时会额外运行一次 memory-prefix context pass。

训练记录 QA、Recon 和 `distill_loss`；`loss` 使用配置权重求和。每次评估先跑 autoregressive QA（EM/F1/ROUGE-L 等答案指标），再跑 teacher-forced recon 指标：`recon_ce`、`recon_ppl`、`recon_token_accuracy`，逐数据集记录。recon 指标只做一次前向、不做任何生成，因此可以放进训练循环；`evaluation.recon_autoregressive: true` 才额外开启昂贵的自回归 recon 评估（每个 context 生成 `max_context_tokens` 个 token）。

checkpoint 只在评估步保存，即 `evaluation.autoregressive_every`，没有独立的 `checkpoint_every` 配置项。

旧 checkpoint 内嵌的旧字段不会自动转换，不能直接用于此版本恢复训练。请保留原代码版本评测历史 checkpoint。

## Context writer

context block 的层编号使用零基索引：`0` 是第一个 decoder block，`-1` 是最后一个 block；不存在 embedding 层编号。中间层 hook 取得 block 输出（尚未经过最终 Qwen norm），`-1` 使用最终 `last_hidden_state`（已经过最终 norm），各层由可学习 softmax 权重混合。Qwen3-1.7B 的窗口上限从实际 backbone config 读取；当前 snapshot 为 40960。

长 context 的 batch size 不能只按固定 memory 长度估算显存：未 padding 的 batch 可走 SDPA 的高效路径，带 padding 的 batch 可能生成较大的 attention mask。`context_gradient_checkpointing` 只重算第一次 context pass 的 block，降低训练显存但增加反向计算时间。

## Readout 模式

`memory.readout_mode` 支持以下模式；默认 `memory_question`，兼容未包含此字段的旧配置。

| 模式 | 输入 readout embeddings 的来源 |
|---|---|
| `memory_question` | Resampler 同时读取 question 和 memory（原行为） |
| `memory_only` | Resampler 只读取 memory，不创建 question projection |
| `learned_prefix` | 全局共享的 `[R, H]` 可学习 embeddings，不读取 question 或 memory 内容 |
| `off` | 不创建 readout 模块、不插入 readout 位置，忽略正数 readout_length |

```yaml
memory:
  readout_mode: memory_only
  readout_length: 8
```

`readout_length: 0` 仍会关闭所有模式的 readout。`readout_layers/heads/hidden_size` 仅对两个 resampler 模式生效。
四种模式保留相同的 memory KV 路径；memory_only/learned_prefix 限制的是插入 Qwen 前的 embeddings 来源，进入 Qwen 后这些位置仍可 attention 到 question 和 memory。
Learned prefix 是等长度控制而非等参数量控制。切换模式用于独立训练的结构消融；不同模式的 checkpoint 参数结构可能不兼容，不应直接作为原 checkpoint 的续训配置。
