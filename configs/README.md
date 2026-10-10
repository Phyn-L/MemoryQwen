# 当前配置

配置直接采用新字段，不提供旧 YAML 字段迁移。机器通过 `--machine 4090/h200/a800` 选择；同一实验不再复制机器版本。

- `train_baseline.yaml`：Qwen3-1.7B、模型窗口 40960、输入过滤上限 8192、memory 64、reader on、LoRA on。writer 默认使用最终层，也可用 `memory.context_layer_indices` 选择多个 block 输出并学习 softmax 混合。
- `ablations/`：每个变量一个配置，文件名说明相对 baseline 的变化：`context_length`、`memory_length`、`multilayer`、`reader_embedding`、`reader_off`、`lora_off` 和 `token_recon`。
- `test.yaml`：一个统一的 checkpoint 评估配置，默认 SQuAD validation。
- `icl_zeroshot.yaml`：一个 zero-shot ICL 配置，可通过 CLI 切换数据集和模型。

## 运行名称

通过 `logging.wandb_run_name: my_experiment` 显式指定运行名称，同时用于 W&B
和 checkpoint 子目录：`<checkpoint.output_dir>/my_experiment/`。
省略该字段、设为 `null` 或空字符串时，自动使用模型名和时间，例如
`Qwen1.7B_20260922_153000`。显式名称应使用单个目录名；相同名称会复用同一路径。
`--resume` 始终保留 checkpoint 所在目录。

## 损失字段

| 字段 | 意义 | 关闭方式 |
|---|---|---|
| `recon_weight` | Recon 任务的 context token CE 权重 | 1 |
| `data.train_tasks` / `data.recon_ratio` | `qa`、`recon` 或混合训练；混合时 Recon 样本比例 | `mix` / 0.5 |
| `token_recon_weight` | MemoryDecoder + vocabulary head 从 memory/位置预测原文 token | 0 |
| `causal_recon_weight` | Qwen 从 memory KV 和真实原文前缀预测下一 token；要求 `M + L <=` backbone window | 0 |
| `causal_recon_positions` | prefix-KV CE 打分位置数；0 为全部 | — |
| `distill_weight` | prefix-KV student 的分布蒸馏；可独立于 prefix CE 开启 | 0 |

训练任务通过 `data.train_tasks` 和 `data.recon_ratio` 控制：

```yaml
data:
  train_tasks: mix   # mix、qa、recon
  recon_ratio: 0.5   # mix 时 Recon 样本比例
```

每个 context 都保留一个 QA 候选和一个 Recon 候选。Recon 使用
`data.recon_prompt`，QA 使用 `data.qa_prompt`；这里采用自然语言 prompt 来区分任务，
不新增 tokenizer special token，避免新增词表参数和 checkpoint 兼容问题。需要先做 Recon
再做 QA 时，把第一阶段 checkpoint 传给 `scripts/train.py --init-from`，第二阶段设
`train_tasks: qa`。

baseline 默认使用 `mix` 训练，其中 QA 和 Recon 各占一半；`token_recon`、causal reconstruction 和蒸馏默认关闭。满 backbone window 的训练应关闭 causal reconstruction 和 distillation，因为它们还需要一次 `M + L` 的 prefix-KV pass。

训练与 teacher-forced 验证记录 QA、Recon、`token_recon_loss`、`causal_recon_loss` 和 `distill_loss`；`loss` 使用配置权重求和。

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
