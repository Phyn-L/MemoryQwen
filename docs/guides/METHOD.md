# Context-to-Memory 方法

`MetaLoRA.encode_context_prefix()` 的数据流固定为两次 backbone 调用：

1. Qwen 对 `[B, L, H]` context 做一次 causal forward。临时 hook 只收集 `memory.context_layer_indices` 指定层；默认 `[-1]`。层状态按 `softmax(context_layer_logits)` 融合。
2. `ContextMemoryResampler` 的 `M` 个 learned queries 对融合后的 `[B, L, H]` 做 cross attention，得到固定长度 `[B, M, H]` memory embedding。
3. Qwen 只对 `[B, M, H]` memory embedding 做短 forward，得到每层 `[B, M, H]` hidden 和 KV cache。QA 通过 `qa_context_indices` 选择对应 cache，因此不会保留 context KV。

`memory.reader_memory_source` 控制 question-conditioned reader 的输入：`memory` 使用最后一层 memory hidden，`memory_embedding` 使用 resampler 原始输出，`layer_mean` 使用所有层平均值。`readout_length: 0` 关闭 reader。

Qwen 的 `max_position_embeddings` 是模型窗口上限。当前 Qwen3-1.7B checkpoint 为 40960；数据加载默认用 `data.max_input_context_length=8192` 过滤超长样本，再由 `max_context_tokens` 控制 batch 截断。writer 接受模型窗口内的非空 context。层编号是零基 decoder block 编号：中间层 hook 取得未经过最终 norm 的 block 输出，`-1` 使用经过最终 norm 的 `last_hidden_state`，不包含 embedding 层。带 context 的 AE pass 还需要 `M + L <=` 模型窗口。

冻结的 backbone 参数和进入 Qwen 的输入保持 checkpoint dtype（通常 bfloat16）；resampler、层融合权重、LoRA 和 decoder 使用 `model.trainable_dtype`（默认 float32）。长 context 可启用 `model.context_gradient_checkpointing: true`；它只作用于第一次 context pass，短 memory pass 和 QA cache pass 不启用 checkpointing，因此反向会重算 context block。

训练支持三种任务组合：`train_tasks: qa`、`train_tasks: recon` 和 `train_tasks: mix`。混合训练通过 `recon_ratio` 选择 Recon 样本；每个 context 同时保留 QA 和 Recon 候选。QA 和 Recon 使用 `qa_prompt`、`recon_prompt` 区分，不新增 special token。固定 memory 是有损压缩，当前冒烟测试只覆盖形状、梯度、padding 隔离和窗口边界，尚未证明下游问答质量。context encoder 的 self-attention 仍然对 `L` 呈二次复杂度；该改造固定的是后续 memory KV 的长度，而不是把任意长度 context 变成线性成本。

使用 `conda activate shine` 后，可用 `python scripts/check_context_memory.py --tiny --device cpu --length 40960` 做结构和窗口冒烟；GPU 上使用 `--device cuda`，多卡使用 `torchrun --nproc_per_node=4`。
