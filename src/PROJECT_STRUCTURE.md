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
| `evaluator.py` | teacher-forced 与 autoregressive 评估，并记录每个 loss/指标。 |
| `metrics.py` | EM、F1、ROUGE-L 等答案指标。 |
| `dtypes.py` | 关闭 autocast 等 dtype 辅助，保证 loss 用 float32 reduction。 |

## 2. 数据处理：JSONL 到训练 batch

入口是 `pipeline.make_context_dataset()`，底层是 `ConversationDataset`：

1. 每个 split 直接读取配置中的 JSONL 文件；每行包含 `context` 和 `conversations`。
2. `ConversationDataset` 建立字节 offset 索引，随机访问时只解析当前行，不把整个文件载入内存。首次扫描会显示 `Indexing conversation data` 进度条，并在 `data.cache_offsets=true` 时缓存过滤后的 offsets；长 context、chat-template Question 和 Answer 的 tokenizer 过滤按 `data.tokenizer_filter_batch_size` 批量执行，当前默认 batch size 为 2048。`max_question_tokens` 限制单个 Question，`max_answer_tokens` 限制单个 Answer，`max_conversation_tokens` 限制一行内所有 Question+Answer 的总长度。缓存放在共享的 `outputs/conversation_cache` 下，并按数据文件、过滤参数和 tokenizer 内容指纹复用，因此兼容模型无需重复处理。空 context、超限 context、Question、Answer 或 conversation，或没有 assistant turn 的行会被过滤。
3. 一个 JSONL 行对应一个 `ContextRecord`。`ContextRecord.turns()` 将每个 assistant 消息配对为一个监督 turn，并保留此前所有 system/user/assistant 消息作为历史。
4. `SortishSampler` 先按 context token 长度排序，再在 bucket 内随机打乱，减少 padding；长度缓存同时绑定数据 fingerprint 和 tokenizer 信息。
5. `collate_context_records` 对 context、conversation prompt、answer 分别 tokenize：
   - `context_ids/context_mask`: `[C, Lc]`
   - `input_ids/input_mask`: `[Q, Lq]`
   - `target_ids/target_mask`: `[Q, La]`
   - `labels`: answer 的有效 token 保留 id，其余位置为 `-100`
   - `qa_context_indices`: `[Q]`，指出每个 QA 属于哪个 context
6. conversation prompt 始终通过 tokenizer 的 `apply_chat_template(..., add_generation_prompt=True)` 构造；训练目标是当前 assistant 内容，因此当前答案不会泄漏到输入，PWC 的历史答案会保留在后续 turn 的输入中。
7. 训练时 `mix` 为每个 context 展开一条 QA 和一条 Recon 行；`qa`/`recon` 只展开对应任务。batch 字段为 `input_ids/input_mask`、`target_ids/target_mask`、`labels`，Recon target 是完整 context token，不追加 EOS。`recon_ratio` 调度只改变 QA/Recon 损失权重；评估只构造 QA。

因此一个 batch 中 context 数 C 和监督行数 Q 可以不同。memory 只对 C 个 context 编码一次，监督行通过 `qa_context_indices` 复用对应的 memory KV。

## 3. memory 的运行过程

### 3.1 写入 memory：context + M 个 slot

`ContextMemoryModel.encode_context_prefix()` 只对 context 做一次 Qwen forward，并通过临时 hook 收集配置的层。softmax 层权重混合这些 hidden 后，`ContextMemoryResampler` 的 M 个 learned queries 跨注意力读取全部 context，输出固定的 memory embeddings。随后只对这 M 个 embedding 做一次短 Qwen forward，形成 `layer_memory [C, num_layers, M, H]` 和可复用的 memory KV cache。Qwen 主干默认冻结，指定 `target_modules` 的线性层插入 LoRA；真正训练的主要是 resampler、层权重、LoRA、decoder、可选 readout/head。

### 3.2 读取 memory：QA、readout 与生成

`forward_qa_with_prefix()` 按 `qa_context_indices` 选择 cache，拼接 `task + question + answer`，并让 question/answer 读取 memory。`task_embeddings[0]` 对应 Recon，`task_embeddings[1]` 对应 QA；Recon 的 question 为空。answer 的 label 前缀全部为 `-100`，所以监督只计算 target token。

当 `readout_length > 0` 时，`QuestionResampler` 根据 task、question 和 memory 生成 R 个 readout latent，序列变为 `task + question + readout(R) + answer`；R 个位置只服务于当前问题，不会改变可缓存的 context memory。生成阶段复用同一 memory cache，按 context 分组解码。

### 3.3 memory embedding 与辅助目标

当前默认模式只由 resampler 输出第 0 层的 `memory_embedding [C,M,H]`。随后将该 embedding 作为 Qwen 的 memory 前缀重新前向，由 Qwen 自身生成所有层的 memory hidden states 和 KV cache；QA、Recon 和蒸馏都复用这份 cache。当前主模型不包含独立的 token reconstruction decoder 分支。

## 4. 训练目标与参数

总目标在 `losses.objective_total()` 中为：

```text
L = task_weighted_mean(L_qa, L_recon)
  + distill_weight * L_distill
```

各项均先按自身有效 token/context 做平均，再乘配置权重。

### 4.1 `qa_loss`（QA 答案交叉熵）

`qa_loss(logits, labels)` 对 logits 做 next-token shift，只看 labels 不为 `-100` 的 answer 位置，并先按每个 QA 行平均，再对 QA 行平均。它是唯一直接对应下游问答的主目标。

- `memory.qa_weight`：QA loss 的总权重，默认 `1.0`。
- `max_answer_tokens`、`append_eos`、`eos_mode`：决定被监督的 answer 序列长度和 EOS 标签。

### 4.2 `Recon`（memory-conditioned context token CE）

训练数据可为每个 context 构造一条 Recon 样本，使用空的任务输入直接生成完整 context。该目标直接检验 memory 是否保留可生成的离散内容。

### 4.3 `distill_loss`（full-context 到 memory-conditioned 的 KL）

teacher 是 encoder 中普通 causal context hidden 产生的分布，student 是带 memory KV 的 autoencoding hidden。两者使用同一 unembedding，计算 `T² * KL(p_teacher || p_student)`；teacher 不反传。它保留 teacher 的软分布，训练 memory 补足主干先验无法确定的 residual information。

- `distill_weight`：总权重；
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
| `qa_per_context`, `sample_qa` | 每个 context 放入 batch 的 QA 数量和采样方式。 |

分析结果时应同时看 QA 指标和 Recon 指标；仅比较总 loss 不足以判断 memory 是否真正存储了 context，因为不同配置可能启用了不同 loss 项。
