# 当前配置

配置直接采用新字段，不提供旧 YAML 字段迁移。机器通过 `--machine 4090/h200/a800` 选择；同一实验不再复制机器版本。

- `train_baseline.yaml`：Qwen3-1.7B、context 1024、memory 64、causal slots、reader on、LoRA on。
- `ablations/`：每个变量一个配置，文件名说明相对 baseline 的变化。
- `test.yaml`：一个统一的 checkpoint 评估配置，默认 SQuAD validation。
- `icl_zeroshot.yaml`：一个 zero-shot ICL 配置，可通过 CLI 切换数据集和模型。

## 损失字段

| 字段 | 意义 | 关闭方式 |
|---|---|---|
| `embedding_recon_weight` | MemoryDecoder 重建原文 input embeddings | 0 |
| `embedding_recon_loss` | `mse`、`cosine`、`mse_cosine` | 由 weight 控制 |
| `embedding_recon_cosine_weight` | MSE+cosine 中 cosine 的系数 | — |
| `token_recon_weight` | MemoryDecoder + vocabulary head 从 memory/位置预测原文 token | 0 |
| `causal_recon_weight` | 原 AE：Qwen 从 memory KV 和真实原文前缀预测下一 token | 0 |
| `causal_recon_positions` | prefix-KV CE 打分位置数；0 为全部 | — |
| `distill_weight` | prefix-KV student 的分布蒸馏；可独立于 prefix CE 开启 | 0 |

baseline 保留原三次训练的目标：memory-token CE=1、prefix-KV CE=1、distill=0.3，embedding recon=0。
表征重建与文本重建可同时启用。关闭的指标记为 0，不表示进行了重建评估。

训练与 teacher-forced 验证分别记录 `embedding_recon_loss`、`token_recon_loss`、`causal_recon_loss`、`distill_loss`，指标均为未加权值。`loss` 使用配置权重求和；QA 按问答行平均，辅助目标按 context 平均，再跨进程汇总。

旧 checkpoint 内嵌的旧字段不会自动转换，不能直接用于此版本恢复训练。请保留原代码版本评测历史 checkpoint。
