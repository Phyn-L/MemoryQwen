# Context-to-Memory 方法

`ContextMemoryModel.encode_context_prefix()` 的数据流固定为两次 backbone 调用：

1. Qwen 对 `[B, L, H]` context 做一次 causal forward。临时 hook 只收集 `memory.context_layer_indices` 指定层；默认 `[-1]`。层状态按 `softmax(context_layer_logits)` 融合。
2. `ContextMemoryResampler` 的 `M` 个 learned queries 对融合后的 `[B, L, H]` 做 cross attention，得到固定长度 `[B, M, H]` memory embedding。
3. Qwen 只对 `[B, M, H]` memory embedding 做短 forward，得到每层 `[B, M, H]` hidden 和 KV cache。QA 通过 `qa_context_indices` 选择对应 cache，因此不会保留 context KV。

reader 只使用一个 Recon 连续任务向量：`recon_task_embedding`，它初始化为 Qwen3 assistant 头 `[<|im_start|>, assistant, \n]`（token id 151644/77091/198）三个预训练输入 embedding 的均值，之后作为普通可训练参数更新；词表不含这些 id 的合成小 backbone（单元测试）退回 `std=0.02` 随机初始化。QA 不插入任务向量；两条路径都直接使用 memory prefix，不新增 tokenizer token：

```text
Recon: [memory, recon_task_embedding] → frozen Qwen → context
QA:    [memory, question] → frozen Qwen → answer
```

QA 将 `[memory, question]` 共同视为 user、answer 视为 assistant；Recon 将 `[memory, recon_task_embedding]` 共同视为 user、context 视为 assistant。训练 batch 的 `task_ids` 仍使用 `0=Recon`、`1=QA` 选择分支。context encoder 的 Qwen forward 使用 MetaLoRA；memory prefix、reader 和 QA generation 使用原始冻结 backbone，不启用 MetaLoRA。

`memory.reader_memory_source` 控制 question-conditioned reader 的输入：`memory_hidden` 使用最后一层 memory hidden，`memory_embedding` 使用 resampler 原始输出，`layer_mean` 使用所有层平均值。`readout_length: 0` 关闭 reader。

Qwen 的 `max_position_embeddings` 是模型窗口，记作 `ContextMemoryModel.window_limit`（当前 Qwen3-1.7B checkpoint 为 40960）。`data.max_context_tokens` 是训练用的 context 上限，构造参数 `max_context_tokens` 会被真正存下来并生效：超过它的 context 直接报错，而 memory + prompt + target 的总长度只受 `window_limit` 约束。writer 接受上限内的非空 context。层编号是零基 decoder block 编号：中间层 hook 取得未经过最终 norm 的 block 输出，`-1` 使用经过最终 norm 的 `last_hidden_state`，不包含 embedding 层。带 context 的 AE pass 还需要 `M + L <=` 模型窗口。

冻结的 backbone 参数和进入 Qwen 的输入保持 checkpoint dtype（通常 bfloat16）；resampler、任务 embedding、层融合权重和 LoRA 使用 `model.trainable_dtype`（默认 float32）。长 context 可启用 `model.context_gradient_checkpointing: true`；它只作用于第一次 context pass，短 memory pass 和 QA cache pass 不启用 checkpointing，因此反向会重算 context block。`model.context_gradient_checkpointing_every_n_layers` 可指定启用间隔，默认 `null` 时自动取 `ceil(sqrt(Qwen层数))`。

训练支持三种任务组合：`train_tasks: qa`、`train_tasks: recon` 和 `train_tasks: mix`。混合训练为每个 context 同时构造 QA 与完整 context Recon 监督，`recon_schedule` 调整损失权重，并由 `recon_ratio_min` 保证 Recon 不会消失。Recon 不输入 question token，只使用 `recon_task_embedding` 和 memory；QA 直接输入 conversation。位置编码只有一个来源：`ContextMemoryModel`/`compact_positions(valid_mask, start)` 按“该 token 在本段有效 token 中的序号”给出绝对位置。context 编码、QA 的 question/read-out/answer 段、Recon 的 task 段和生成路径都调用它，因此同一段 token 在训练、编码与生成中的编号一致，而且与 padding 方向无关（左 padding 与右 padding 得到同一份 memory；旧实现用 `arange`，只在右 padding 下正确）。

损失实现只有一套：`losses.sequence_lm_loss` 把整批 `[rows, tokens, hidden]` flatten 成一次词表投影（按 `loss.max_logits_rows` 分块），用行索引 `index_add_` 做行内归一化，空行由 `supervision_mask` 排除，因此“行内先平均、再按任务平均”的语义不依赖逐行 Python 循环。蒸馏的位置采样由训练循环传入的 `torch.Generator`（按 `training.seed + step` 派生）决定，断点续跑与原运行一致。

评估阶段除了 autoregressive QA，还跑一次 teacher-forced recon 指标（`Evaluator.recon_teacher_forced`）：它复用 Recon 目标的同一次前向，直接给出 `recon_ce` / `recon_ppl` / `recon_token_accuracy`，不做任何解码，因此可以看到 memory 到底保留了多少 context。

固定 memory 是有损压缩，当前冒烟测试只覆盖形状、梯度、padding 隔离和窗口边界，尚未证明下游问答质量。context encoder 的 self-attention 仍然对 `L` 呈二次复杂度；该改造固定的是后续 memory KV 的长度，而不是把任意长度 context 变成线性成本。

使用 `conda activate shine` 后，可用 `python scripts/check_context_memory.py --tiny --device cpu --length 64 --backward` 做结构、损失与反向冒烟（tiny 模型用 `max_position_embeddings` 当上限，可以给很大的 `--length`）；GPU 上使用 `--device cuda`，多卡使用 `torchrun --nproc_per_node=4`。真模型上 `--length` 不能超过 `data.max_context_tokens`（baseline 为 1536，`--length 40960` 现在会按预期报错）。`--backward` 会同时跑 QA 与 Recon 两条监督，因此 `recon_task_embedding` 也会拿到梯度。
