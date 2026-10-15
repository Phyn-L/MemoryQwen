# MemoryQwen `src` 项目结构与训练目标

本文档按当前源码说明数据流、memory 的组成和训练目标。配置字段以 `utils/config.py` 的 `MemoryConfig`、`DataConfig` 为准；实际运行时还要结合所选 YAML（例如 `configs/train_baseline.yaml`），因为 dataclass 默认值和实验配置可能不同。

## 1. `src` 中各模块的职责

| 文件 | 作用 |
| --- | --- |
| `data.py` | 读取聚合后的 JSONL，按输入长度过滤/缓存 context，构造 QA/Recon 任务，tokenize，并生成 batch。 |
| `cache_utils.py` | offset 和 sortish 长度缓存共用的指纹与文件锁工具。 |
| `pipeline.py` | 根据配置创建 train/validation/test dataset、collator 和 sortish sampler。 |
| `model.py` | 冻结 Qwen 主干，写入 memory slot，输出第 0 层 memory embedding，并由 Qwen 前向构造各层 memory KV，执行 QA、Recon 和蒸馏前向。 |
| `resampler.py` | 可选的 question-conditioned readout，把 memory 变成插入 QA 序列的 R 个 latent。 |
| `losses.py` | QA CE、Recon CE 和 KL distillation 的实现与加权求和。 |
| `evaluator.py` | autoregressive 生成式评估，记录答案指标及各数据集指标。 |
| `metrics.py` | EM、F1、ROUGE-L 等答案指标。 |
| `dtypes.py` | 关闭 autocast 等 dtype 辅助，保证 loss 用 float32 reduction。 |

## 2. 数据处理：JSONL 到训练 batch

入口是 `pipeline.make_context_dataset()`，底层是 `ConversationDataset`：

1. 每个 split 读取配置解析出的 JSONL：`train`/`validation` 默认是**过滤后的 cache** `<data.files.root>/1qa/cache/{train,validation}.jsonl`，`test` 默认是原始 `<data.files.root>/1qa/meta/meta_test.json`（没有过滤后的 test 产物），都可用 `<split>_conversation_file` 覆盖；每行包含 `context` 与 `conversations` 或 `qa_pairs`。原始数据在 `<data.files.root>/1qa/meta/meta_<split>.json`（`scripts/data/build_1qa.py`，不依赖配置），cache 由 `scripts/data/build_cache.py` 按当前配置过滤生成（`scripts/train.py` 启动时自动补建/重建，`src/cache_build.py` 是规则所在）。loader 会先核对 cache 指纹（`pipeline.uses_filtered_cache`）：一致就设 `ConversationDataset(pre_filtered=True)` 直接读，不再重复 tokenize；不一致/被 `--data-file` 覆盖时才走逐行过滤并另存过滤缓存。
2. `ConversationDataset` 会把过滤后的行缓存到 loader 预处理目录（默认 `<data.root>/1qa/cache/loader`，可用 `data.preprocess_cache_dir` 指向本地快盘；排序长度缓存也放在这里；若 cache 由 `build_cache.py` 构建，长度已在过滤时记进 `cache/<split>.jsonl.lengths.json`，这一遍会被跳过）；`data.cache_filtered_dataset` 控制是否启用该缓存。**当读的就是刚构建的 `cache/train.jsonl`/`cache/validation.jsonl` 且指纹匹配时，这层过滤与缓存都不会发生**（`pipeline.uses_filtered_cache`）。长 context、chat-template Question 和 Answer 的 tokenizer 过滤按 `data.tokenizer_filter_batch_size` 批量执行，当前默认 batch size 为 2048。`max_question_tokens` 限制单个 Question，`max_answer_tokens` 限制单个 Answer，`max_conversation_tokens` 限制一行内所有 Question+Answer 的总长度。缓存按数据文件、过滤参数和 tokenizer 内容指纹复用，因此兼容模型无需重复处理。空 context、超限 context、Question、Answer 或 conversation，或没有 assistant turn 的行会被过滤。
3. 一个 JSONL 行对应一个 `ContextRecord`。`ContextRecord.turns()` 将每个 assistant 消息配对为一个监督 turn，并保留此前所有 system/user/assistant 消息作为历史。
4. `SortishSampler` 先按 context token 长度排序，再在 bucket 内随机打乱，减少 padding；长度缓存同时绑定数据 fingerprint 和 tokenizer 信息。
5. `collate_context_records` 对 context、conversation prompt、answer 分别 tokenize：
   - `context_ids/context_mask`: `[C, Lc]`
   - `input_ids/input_mask`: `[Q, Lq]`
   - `target_ids/target_mask`: `[Q, La]`
   - `labels`: answer 的有效 token 保留 id，其余位置为 `-100`
   - `qa_context_indices`: `[Q]`，指出每个 QA 属于哪个 context
6. conversation prompt 始终通过 tokenizer 的 `apply_chat_template(..., add_generation_prompt=True)` 构造；训练目标是当前 assistant 内容，因此当前答案不会泄漏到输入，PWC 的历史答案会保留在后续 turn 的输入中。
7. 训练时 `mix` 为每个 context 展开一条 QA 和一条 Recon 行；`qa`/`recon` 只展开对应任务。batch 字段为 `input_ids/input_mask`、`target_ids/target_mask`、`labels`；Recon target 是完整 context token。`data.append_eos` 会给每一条监督行追加 EOS（QA 与 Recon 一致），`recon_ratio` 调度只改变 QA/Recon 损失权重。`mix` 下每个 context 的 QA turn 由 `_turn_rng(record, epoch)` 确定性抽取，因此与 DataLoader worker 数量无关。

因此一个 batch 中 context 数 C 和监督行数 Q 可以不同。memory 只对 C 个 context 编码一次，监督行通过 `qa_context_indices` 复用对应的 memory KV。

## 3. memory 的运行过程

### 3.1 写入 memory：context + M 个 slot

`ContextMemoryModel.encode_context_prefix()` 只对 context 做一次 Qwen forward，并通过临时 hook 收集配置的层。softmax 层权重混合这些 hidden 后，`ContextMemoryResampler` 的 M 个 learned queries 跨注意力读取全部 context，输出固定的 memory embeddings。随后只对这 M 个 embedding 做一次短 Qwen forward，形成 `layer_memory [C, num_layers, M, H]` 和可复用的 memory KV cache。Qwen 主干默认冻结，指定 `target_modules` 的线性层插入 LoRA；真正训练的主要是 resampler、层权重、LoRA、decoder、可选 readout/head。

### 3.2 读取 memory：QA、readout 与生成

`forward_with_memory_prefix()` 按 `qa_context_indices` 选择 cache。QA 拼接 `question + answer`，Recon 拼接 `recon_task_embedding + context`；两条路径都让 user 段读取 memory，answer/context 作为 assistant 目标。answer 的 label 前缀全部为 `-100`，所以监督只计算 target token。

当 `readout_length > 0` 时，`QuestionResampler` 根据 task、question 和 memory 生成 R 个 readout latent，序列变为 `task + question + readout(R) + answer`；R 个位置只服务于当前问题，不会改变可缓存的 context memory。生成阶段复用同一 memory cache，按 context 分组解码。

### 3.3 memory embedding 与辅助目标

当前默认模式只由 resampler 输出第 0 层的 `memory_embedding [C,M,H]`。随后将该 embedding 作为 Qwen 的 memory 前缀重新前向，由 Qwen 自身生成所有层的 memory hidden states 和 KV cache；QA、Recon 和蒸馏都复用这份 cache。当前主模型不包含独立的 token reconstruction decoder 分支。

## 4. 训练目标与参数

总目标在 `losses.objective_total()` 中为：

```text
L = task_weighted_mean(L_qa, L_recon)
  + distill_weight * L_distill
```

各项均先按自身有效 token/context 做平均，再乘配置权重。实现上整批的 `[rows, tokens, hidden]` 被 flatten 成一次词表投影（按 `loss.max_logits_rows` 分块），用行索引 `index_add_` 做行内归一化，再用 `supervision_mask` 排除没有目标的空行；因此每行先平均、再按任务平均的语义不依赖逐行 Python 循环。

### 4.1 `qa_loss`（QA 答案交叉熵）

`qa_loss(logits, labels)` 对 logits 做 next-token shift，只看 labels 不为 `-100` 的 answer 位置，并先按每个 QA 行平均，再对 QA 行平均。它是唯一直接对应下游问答的主目标。

- `loss.qa_weight` / `loss.recon_weight`：QA 与 Recon 的总权重，默认 `1.0`。
- `max_answer_tokens`、`append_eos`、`eos_mode`：决定被监督的 answer 序列长度和 EOS 标签。

### 4.2 `Recon`（memory-conditioned context token CE）

训练数据可为每个 context 构造一条 Recon 样本：`[memory, recon_task_embedding]` 作为 user 段、完整 context 作为 assistant 目标，监督下一个 token。该目标直接检验 memory 是否保留可生成的离散内容。

评估阶段有对应的 teacher-forced 指标（`Evaluator.recon_teacher_forced`）：同一次前向的 hidden 直接算 context token 的交叉熵与逐 token 精确率，输出 `recon_ce` / `recon_ppl` / `recon_token_accuracy`，不需要解码。两者共用 `losses.teacher_forced_row_stats`，因此训练 loss 与评估指标恒等（逐行 nll/tokens）。

### 4.3 `distill_loss`（full-context 到 memory-conditioned 的 KL）

teacher 是 encoder 中普通 causal context hidden 产生的分布，student 是带 memory KV 的 autoencoding hidden。两者使用同一 unembedding，计算 `T² * KL(p_teacher || p_student)`；teacher 不反传。它保留 teacher 的软分布，训练 memory 补足主干先验无法确定的 residual information。

- `loss.distill_weight`：总权重；
- `distill_temperature`：温度 T，必须大于 0；
- `distill_positions`：采样位置数，`<=0` 表示全部；
- `distill_topk`：teacher 只保留 top-k token，0 表示不截断；
- `distill_entropy_weight`：是否按 teacher entropy 对位置加权。

## 5. 常用结构参数的含义

| 参数 | 含义 |
| --- | --- |
| `memory_length` | memory slot 数 M，决定压缩 bottleneck 长度。 |
| `context_layer_indices` | context 编码中参与 softmax 融合的 Qwen 层，`-1` 表示最后一层。 |
| `memory_resampler_*` | learned queries 的数量、深度、头数和 bottleneck 宽度。 |
| `lora_rank`, `lora_alpha`, `target_modules` | 主干 LoRA 的秩、缩放和插入层。 |
| `readout_length` | 每个问题额外生成的 readout latent 数 R；0 为关闭。 |
| `readout_layers`, `readout_heads`, `readout_hidden_size` | readout Transformer 的深度、头数和宽度。 |
| `max_context_tokens`, `max_question_tokens`, `max_answer_tokens`, `max_conversation_tokens` | context、单个 Question、单个 Answer，以及一行内所有 Question+Answer 的总长度上限。 |
| `qa_per_context`, `sample_qa` | 每个 context 放入 batch 的 QA 数量与抽取方式（训练按 `(record, epoch)` 确定性抽取；评估使用按数据集分层的全局种子抽样）。 |
| `training.gradient_accumulation_steps` | 每个优化器步累积的 micro-batch 数；调度、评估与 checkpoint 间隔都以优化器步计。 |
| `loss.max_logits_rows` | 一次词表投影的行数上限；整批被 flatten 成一次投影，行内先按自身 token 数平均、再按任务平均。 |
| `data.append_eos`, `data.question_padding_side` | 监督行是否追加 EOS；question 段的 padding 方向（位置由 `compact_positions` 统一推导，与 padding 方向无关）。 |

分析结果时应同时看 QA 指标和 Recon 指标；仅比较总 loss 不足以判断 memory 是否真正存储了 context，因为不同配置可能启用了不同 loss 项。
