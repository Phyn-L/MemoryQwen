# MemoryQwen `src` 项目结构与训练目标

本文档按当前源码说明数据流、memory 的组成和五个训练 loss。配置字段以 `utils/config.py::MemoryConfig` 为准；实际运行时还要结合所选 YAML（例如 `configs/train_baseline.yaml`），因为 dataclass 默认值和实验配置可能不同。

## 1. `src` 中各模块的职责

| 文件 | 作用 |
| --- | --- |
| `data.py` | 读取聚合后的 JSONL，过滤/缓存 context，采样 QA，tokenize，并生成 batch。 |
| `dataset_cache.py` | Hugging Face Dataset 与 sortish 长度缓存；用文件指纹避免旧缓存误用。 |
| `pipeline.py` | 根据配置创建 train/validation/test dataset、collator 和 sortish sampler。 |
| `model.py` | 冻结 Qwen 主干，写入 memory slot，构造每层 memory KV，执行 QA、重构和蒸馏前向。 |
| `MemoryDecoder.py` | 将每层 memory hidden 映射为 context embedding 或 token-reconstruction hidden。 |
| `resampler.py` | 可选的 question-conditioned readout，把 memory 变成插入 QA 序列的 R 个 latent。 |
| `losses.py` | QA CE、embedding/token/causal reconstruction 和 KL distillation 的实现与加权求和。 |
| `evaluator.py` | teacher-forced 与 autoregressive 评估，并记录每个 loss/指标。 |
| `metrics.py` | EM、F1、ROUGE-L 等答案指标。 |
| `dtypes.py` | 关闭 autocast 等 dtype 辅助，保证 loss 用 float32 reduction。 |

## 2. 数据处理：JSONL 到训练 batch

入口是 `pipeline.make_context_dataset()`，底层是 `AggregatedContextDataset`：

1. 在 `data.root/<dataset>/<split>.jsonl` 查找文件；`dataset="all"` 时读取所有子目录。
2. 每行读取 `context` 和 `qa_pairs`。空 context、超过 `max_context_tokens` 的 context，或没有有效 question/answer 的行会被过滤。
3. 一个 JSONL 行对应一个 `ContextRecord`，其中包含一个 context 和多个 `QARecord`。所有非空 `answers` 会保留；训练使用第一个答案，评估可使用全部 reference。
4. 可选 `source_version` 只保留匹配版本的 QA。构建结果写入 HF Dataset cache；文件大小、mtime、过滤条件等组成 fingerprint，避免数据或 tokenizer 配置变化后错误复用缓存。
5. `SortishSampler` 先按 context token 长度排序，再在 bucket 内随机打乱，减少 padding；长度缓存同时绑定数据 fingerprint 和 tokenizer 信息。
6. `collate_context_records` 对 context、question、answer 分别 tokenize：
   - `context_ids/context_mask`: `[C, Lc]`
   - `question_ids/question_mask`: `[Q, Lq]`
   - `answer_ids/answer_mask`: `[Q, La]`
   - `labels`: answer 的有效 token 保留 id，其余位置为 `-100`
   - `qa_context_indices`: `[Q]`，指出每个 QA 属于哪个 context
7. 每个 context 随机采样最多 `qa_per_context` 个 QA（`sample_qa=false` 时全取）。answer 可追加 EOS；`eos_mode="append"` 会扩展一列，避免覆盖最后一个真实 token。

因此一个 batch 中 context 数 C 和 QA 行数 Q 可以不同。memory 只对 C 个 context 编码一次，QA 行通过 `qa_context_indices` 复用对应的 memory KV。

## 3. memory 的运行过程

### 3.1 写入 memory：context + M 个 slot

`MetaLoRA.encode_context_prefix()` 先把 context embedding 与可学习的 `memory_tokens` 拼接：`[context (Lc), memory slots (M)]`。Qwen 主干默认冻结，指定 `target_modules` 的线性层插入 LoRA；真正训练的主要是 memory slot、LoRA、decoder、可选 readout/head。

`build_block_causal_mask` 控制 slot 如何读取 context 及彼此关系：

- `isolated`：每个 slot 都读取有效 context，slot 之间不互读；
- `causal`：slot 可读取更早 slot（含自身）；
- `bidirectional`：slot 全互读。

Qwen 输出每层 hidden state 和 KV cache。截取每层 context 之后的 M 个位置，形成 `layer_memory [C, num_layers, M, H]` 和 `memory_cache`。最后一层 slot hidden 是 QA 前缀表示；每层 KV 则用于后续 continuation。context hidden 同时保留作 distillation teacher。

### 3.2 读取 memory：QA、readout 与生成

`forward_qa_with_prefix()` 按 `qa_context_indices` 选择 cache，拼接 `question + answer`，并让 question/answer 读取 memory。answer 的 label 前缀全部为 `-100`，所以 QA loss 只计算 answer token。

当 `readout_length > 0` 时，`QuestionResampler` 根据 question 和 memory 生成 R 个 readout latent，序列变为 `question + readout(R) + answer`；R 个位置只服务于当前问题，不会改变可缓存的 context memory。生成阶段复用同一 memory cache，按 context 分组解码。

### 3.3 memory-only 重构分支

每层 `MemoryDecoder` 只接收该层的 memory hidden 和位置查询，不读取原 context token，因此 token reconstruction 能检验信息是否真的进入 bottleneck。另一条 causal reconstruction 路径把 memory KV 作为前缀，再用冻结 Qwen 对 context 做 teacher-forced 解码；该路径允许读取真实的历史 context token，主要衡量 memory 对主干预测分布的影响。

## 4. 五个 loss 与参数

总目标在 `losses.objective_total()` 中为：

```text
L = qa_weight * L_qa
  + embedding_recon_weight * L_emb
  + token_recon_weight * L_tok
  + causal_recon_weight * L_causal
  + distill_weight * L_distill
```

各项均先按自身有效 token/context 做平均，再乘配置权重。

### 4.1 `qa_loss`（QA 答案交叉熵）

`qa_loss(logits, labels)` 对 logits 做 next-token shift，只看 labels 不为 `-100` 的 answer 位置，并先按每个 QA 行平均，再对 QA 行平均。它是唯一直接对应下游问答的主目标。

- `memory.qa_weight`：QA loss 的总权重，默认 `1.0`。
- `max_answer_tokens`、`append_eos`、`eos_mode`：决定被监督的 answer 序列长度和 EOS 标签。

### 4.2 `embedding_recon_loss`（连续 embedding 重构）

`MemoryDecoder` 输出 `[C, layers, Lc, H]`，目标是 context 输入 embedding。可选 `mse`、`cosine`、`mse_cosine`（即 `MSE + embedding_recon_cosine_weight * cosine`）。

相关参数：`embedding_recon_weight` 控制总项；`embedding_recon_cosine_weight` 控制 cosine 子项；`decoder_hidden_size`、`decoder_heads`、`decoder_ffn_ratio` 定义 decoder 容量。该目标容易学到平均向量，因此单独下降不能证明 memory 能恢复具体 token。

### 4.3 `token_recon_loss`（memory-only token CE）

每个 layer decoder 从 memory hidden 和位置 query 直接预测 context token，不能读取其他 context token；使用共享 vocabulary head，按 layer 平均 CE。它比 embedding MSE 更直接地要求 memory 保存离散内容。

- `token_recon_weight`：总权重；
- `token_recon_positions`：每个 context 每步采样的目标位置数，`<=0` 表示全部；
- `head_mode`：`linear` 从头学习 `[vocab, decoder_hidden]`，`tied` 使用冻结 Qwen embedding 加小 adapter；
- `head_init`：tied head 的 `auto`、`random` 或 `memory_projection` 初始化；
- `token_recon`：模型侧是否创建 decoder/head（由训练配置映射）。

### 4.4 `causal_recon_loss`（memory-prefixed causal LM CE）

将 memory KV 作为 Qwen 前缀，对 context 的下一 token 做 teacher-forced CE：`P(t_i | memory, t_<i)`。它使用冻结主干自己的 tied unembedding，context 历史 token 可见，因此更像压缩前缀对语言模型预测的恢复能力，而非严格 memory-only 探针。

- `causal_recon_weight`：总权重，0 表示关闭；
- `causal_recon_positions`：采样位置数，`<=0` 表示所有有效位置；
- `ae_lm`：模型是否启用 memory-prefixed autoencoding 路径。

### 4.5 `distill_loss`（full-context 到 memory-conditioned 的 KL）

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
| `slot_attention` | slot 间 isolated/causal/bidirectional 连接方式。 |
| `lora_rank`, `lora_alpha`, `target_modules` | 主干 LoRA 的秩、缩放和插入层。 |
| `readout_length` | 每个问题额外生成的 readout latent 数 R；0 为关闭。 |
| `readout_layers`, `readout_heads`, `readout_hidden_size` | readout Transformer 的深度、头数和宽度。 |
| `max_context_tokens`, `max_question_tokens`, `max_answer_tokens` | 三类序列的截断上限。 |
| `qa_per_context`, `sample_qa` | 每个 context 放入 batch 的 QA 数量和采样方式。 |

分析结果时应同时看 QA 指标和 memory-only 指标；仅比较总 loss 不足以判断 memory 是否真正存储了 context，因为不同配置可能启用了不同 loss 项。
