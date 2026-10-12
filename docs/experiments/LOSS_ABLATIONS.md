# Loss ablations：配置与最佳 AR 结果

本次分析范围：此前重命名的 **15 个固定 run**（02、04、05–17），不是项目全部历史实验。数据于 2026-09-21 通过 W&B Public API 的 `scan_history()` 全量读取；15 个 run 均为 finished。分析对象是训练期间的 validation 指标，不是独立 test 结果。

## 结论与统计口径

当前候选第一名是 **11：QA + D(0.3)，teacher entropy weighting 开启**，best AR F1 **69.4611**（step **3200**），同一步 AR EM **55.9570**、ROUGE-L **69.1772**。相对同阶段 QA-only（10）的 best AR F1 提升 **1.4873 个百分点**。这是单次训练上的候选优势，尚不能视为跨 seed 的确定增益。

- 主排序指标：`max_t val_autoregressive/f1(t)`；不使用 final score 或 W&B summary 的末值排名。
- 主表的 EM、ROUGE-L 取 **best AR F1 的同一 step**，保证对应一个可选择的 checkpoint；另列各指标独立 best，不能把它们拼成一个 checkpoint 的表现。
- 所有分数均乘 100，差值单位为百分点（pp）。并列 best 取历史中首次达到的 step。
- TF F1 使用独立历史 best，仅作诊断，不参与排名，也不代表与 AR best 同一步。
- 当前代码按 AR F1 保存 checkpoint（`scripts/train.py`）；本次未读取远端 checkpoint 文件来确认 best 权重文件是否仍保留。
- 02/04/05–09 各有 12 次 AR 评估、末步 4509；10–17 各有 11 次、末步 4208。跨阶段的搜索机会和训练步数也不相同。

## 配置矩阵

总损失为 `L = L_QA + E*L_embedding + T*L_token + C*L_causal + D*L_distill`，所有 run 的 QA 权重为 1。E/T/C/D 是 YAML/W&B config 中的数值权重，不是评测分数。

`entropy` 对应 `memory.distill_entropy_weight`：按 teacher 预测分布的熵对各位置蒸馏项加权（见 `src/losses.py`），不是一个额外独立损失。D=0 时该开关不生效。

纠正此前聊天表格：04 与 06 的原始 entropy 配置均为 **true**，不能因其不生效就写成 false；04 实际是 **QA + embedding reconstruction**，目录 `qa_only` 是历史误命名。另，此前声称读取了本地 `outputs/ab_loss_terms` 实际目录不准确：本地没有该目录，本次目录标签依据远程 run 的 `checkpoint.output_dir`，配置依据远程保存的实际 config，并参考仓库生成 YAML 的脚本；未声称读取了远端原始 YAML 文件。

| 实验目录标签 | Run ID | E | T | C | D | entropy 原值 | entropy 生效 | batch size |
|---|---|---:|---:|---:|---:|---|---|---:|
| 02_no_causal_recon | [trrmddtk](https://wandb.ai/34520222200995-xiamen-university/MemoryQwen/runs/trrmddtk) | 1 | 1 | 0 | 0.3 | true | 开启 | 8 |
| 04_qa_only | [o9n3472o](https://wandb.ai/34520222200995-xiamen-university/MemoryQwen/runs/o9n3472o) | 1 | 0 | 0 | 0 | true | 不生效（D=0） | 8 |
| 05_full_no_embedding | [d76ifv0v](https://wandb.ai/34520222200995-xiamen-university/MemoryQwen/runs/d76ifv0v) | 0 | 1 | 1 | 0.3 | true | 开启 | 8 |
| 06_qa_only_true | [573vzh2t](https://wandb.ai/34520222200995-xiamen-university/MemoryQwen/runs/573vzh2t) | 0 | 0 | 0 | 0 | true | 不生效（D=0） | 8 |
| 07_token_only | [64zew61n](https://wandb.ai/34520222200995-xiamen-university/MemoryQwen/runs/64zew61n) | 0 | 1 | 0 | 0 | true | 不生效（D=0） | 8 |
| 08_causal_only | [sse0kt2u](https://wandb.ai/34520222200995-xiamen-university/MemoryQwen/runs/sse0kt2u) | 0 | 0 | 1 | 0 | true | 不生效（D=0） | 8 |
| 09_distill_only | [4ozbj4sa](https://wandb.ai/34520222200995-xiamen-university/MemoryQwen/runs/4ozbj4sa) | 0 | 0 | 0 | 0.3 | true | 开启 | 8 |
| 10_qa_only_control | [ejvgkejq](https://wandb.ai/34520222200995-xiamen-university/MemoryQwen/runs/ejvgkejq) | 0 | 0 | 0 | 0 | false | 不生效（D=0） | 10 |
| 11_distill03_entropy_true | [4xhnoqoe](https://wandb.ai/34520222200995-xiamen-university/MemoryQwen/runs/4xhnoqoe) | 0 | 0 | 0 | 0.3 | true | 开启 | 10 |
| 12_distill03_entropy_false | [ml7ojn3i](https://wandb.ai/34520222200995-xiamen-university/MemoryQwen/runs/ml7ojn3i) | 0 | 0 | 0 | 0.3 | false | 关闭 | 10 |
| 13_distill01_entropy_true | [p58y7w9m](https://wandb.ai/34520222200995-xiamen-university/MemoryQwen/runs/p58y7w9m) | 0 | 0 | 0 | 0.1 | true | 开启 | 10 |
| 14_distill01_entropy_false | [7rm0v4da](https://wandb.ai/34520222200995-xiamen-university/MemoryQwen/runs/7rm0v4da) | 0 | 0 | 0 | 0.1 | false | 关闭 | 10 |
| 15_token_distill_entropy_true | [tyfxwkgk](https://wandb.ai/34520222200995-xiamen-university/MemoryQwen/runs/tyfxwkgk) | 0 | 1 | 0 | 0.3 | true | 开启 | 10 |
| 16_token_distill_entropy_false | [i9lilsh8](https://wandb.ai/34520222200995-xiamen-university/MemoryQwen/runs/i9lilsh8) | 0 | 1 | 0 | 0.3 | false | 关闭 | 10 |
| 17_causal_distill_entropy_true | [7yvpklkf](https://wandb.ai/34520222200995-xiamen-university/MemoryQwen/runs/7yvpklkf) | 0 | 0 | 1 | 0.3 | true | 开启 | 10 |

名称格式为 `实验目录标签__run_id`。所有 `*_only` 均指“仅此辅助项”，QA 始终存在。

## 共同配置与可比性

逐字段展开比较 15 个实际 run config：除 logging/checkpoint 记录外，变化字段仅有 E/T/C/D、entropy 和 `training.batch_size`。已记录配置一致不等于代码、数据内容及多卡运行环境均一致。

| 项目 | 共同配置 |
|---|---|
| Backbone / machine | Qwen3-1.7B / h200 |
| LoRA | rank=8，alpha=16，dropout=0；q/k/v/o + gate/up/down projections |
| Context / memory / reader | context 上限 1024；memory=64；readout=8、2 层、hidden=256、heads=4 |
| Slot / memory 初始化 | causal；token_embed，init_seed=0 |
| Decoder / head | hidden=256，heads=8，FFN ratio=2；tied head，head_init=auto |
| 辅助损失 | token/causal/distill positions 均为 128；distill temperature=1、topk=0 |
| Embedding loss | mse_cosine，cosine weight=0.5（仅 E>0 时生效） |
| 优化 | AdamW，lr=1e-4，weight_decay=0.01；cosine scheduler，warmup=200 steps |
| 训练 | seed=42，epochs=1，qa_per_context=4，train_datasets=all，无 train_max_samples 限制 |
| 验证 | SQuAD validation，validation_max_samples=2000；AR max_qa=1024 |
| 生成 | max_new_tokens=32，qa_batch_size=4；无 chat template |
| 评估频率 | TF 每 200 steps，AR 每 400 steps，另含末步评估 |

`use_peft=false` 不能作为这些历史 run 的 LoRA-off 标签；前次错误标签已删除。这里记录用户确认的 LoRA-on 状态及非空 target_modules，不把旧字段当作开关。

batch size 为 config 中的值；没有从记录中核实 world_size/实际全局 batch、数据版本或代码 commit。阶段内比较更可靠；跨阶段只能作探索性比较，尤其不要把 09 与 11 当成相同训练条件的重复种子。

## 主结果：按 best AR F1 排序

EM@best 与 RL@best 均与 AR F1 最优点同一步；TF best 独立取最大。

| 排名 | 实验 | best AR F1 | step | AR EM@best | AR RL@best | TF best F1（参考） | AR 评估次数 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 11_distill03_entropy_true | 69.4611 | 3200 | 55.9570 | 69.1772 | 69.6548 | 11 |
| 2 | 05_full_no_embedding | 68.4032 | 3600 | 55.8594 | 68.0980 | 68.8736 | 12 |
| 3 | 16_token_distill_entropy_false | 68.2901 | 4208 | 54.4922 | 67.9570 | 68.9243 | 11 |
| 4 | 09_distill_only | 68.2250 | 4400 | 55.2734 | 67.8511 | 69.7032 | 12 |
| 5 | 06_qa_only_true | 68.0169 | 4000 | 54.2969 | 67.8120 | 69.9576 | 12 |
| 6 | 10_qa_only_control | 67.9739 | 4208 | 54.9805 | 67.5633 | 69.4128 | 11 |
| 7 | 02_no_causal_recon | 67.8622 | 4509 | 53.5156 | 67.5109 | 69.7855 | 12 |
| 8 | 14_distill01_entropy_false | 67.6109 | 4000 | 54.8828 | 67.2489 | 69.0400 | 11 |
| 9 | 12_distill03_entropy_false | 67.1298 | 3200 | 53.0273 | 66.9764 | 69.3125 | 11 |
| 10 | 07_token_only | 67.1270 | 4400 | 54.1016 | 66.6999 | 68.2425 | 12 |
| 11 | 15_token_distill_entropy_true | 66.9049 | 4208 | 53.8086 | 66.5815 | 68.6465 | 11 |
| 12 | 13_distill01_entropy_true | 66.7858 | 4208 | 52.8320 | 66.5750 | 69.6334 | 11 |
| 13 | 08_causal_only | 66.5862 | 4400 | 52.2461 | 66.3971 | 69.0979 | 12 |
| 14 | 17_causal_distill_entropy_true | 66.2901 | 4000 | 52.9297 | 65.9437 | 69.1443 | 11 |
| 15 | 04_qa_only | 65.3930 | 4000 | 52.9297 | 65.2304 | 67.5232 | 12 |

## 各 AR 指标独立 best（补充）

下表允许 EM 与 ROUGE-L 在不同 step 达到最优；不用于替换主表的 checkpoint 选择规则。

| 实验编号 | best AR EM | EM step | best AR ROUGE-L | ROUGE-L step |
|---|---:|---:|---:|---:|
| 02 | 53.6133 | 4400 | 67.5109 | 4509 |
| 04 | 53.1250 | 4509 | 65.2304 | 4000 |
| 05 | 55.8594 | 3600 | 68.0980 | 3600 |
| 06 | 54.2969 | 4000 | 67.8120 | 4000 |
| 07 | 54.1016 | 4400 | 66.6999 | 4400 |
| 08 | 52.6367 | 2400 | 66.3971 | 4400 |
| 09 | 55.6641 | 4509 | 67.8511 | 4400 |
| 10 | 55.3711 | 4000 | 67.5633 | 4208 |
| 11 | 55.9570 | 3200 | 69.1772 | 3200 |
| 12 | 53.1250 | 4208 | 66.9764 | 3200 |
| 13 | 53.6133 | 2800 | 66.5750 | 4208 |
| 14 | 54.8828 | 4000 | 67.2489 | 4000 |
| 15 | 53.8086 | 4208 | 66.5815 | 4208 |
| 16 | 54.4922 | 4208 | 67.9570 | 4208 |
| 17 | 53.1250 | 3200 | 65.9437 | 4000 |

## 对照分析：以 AR 为核心

### 1. Phase 2：辅助目标并非越多越好

以下使用 batch=8 的 06（QA-only，68.0169）为参考，比较各自 best AR F1：

| 处理 | 对照 | Δ best AR F1（pp） | 观察 |
|---|---|---:|---|
| 05：T+C+D | 06：QA-only | +0.3863 | 完整无 embedding 组合只有小幅优势 |
| 09：D(0.3)，entropy=true | 06 | +0.2081 | 单独蒸馏只有小幅优势 |
| 07：T | 06 | −0.8899 | 单独 historical token reconstruction ablation 未显示收益 |
| 08：C | 06 | −1.4307 | 单独 causal reconstruction 未显示收益 |
| 04：E | 06 | −2.6239 | QA 背景中 embedding reconstruction 较差；跨执行批次，需核实代码/数据一致性 |

上述结果不支持“historical token reconstruction ablation 总是有用”。它是否有效取决于其他损失、权重和蒸馏加权方式。04 的下降也不能直接外推到所有 E 背景；判断完整组合中 E 的边际贡献需要 matched 00 vs 05，本次 15-run 范围不含 00。

### 2. Phase 3：entropy 的收益取决于蒸馏权重与辅助项

batch=10 的 QA-only 10 为 67.9739。11 为 69.4611（+1.4873 pp），16 为 68.2901（+0.3162 pp）；其余本次 phase-3 候选的 best AR F1 均低于 10。

| 处理 | 对照 | Δ best AR F1（pp） | 被检验的因素 |
|---|---|---:|---|
| 11 | 12 | +2.3313 | D=0.3、无 T/C 时，开启 entropy |
| 13 | 14 | −0.8251 | D=0.1、无 T/C 时，开启 entropy |
| 15 | 16 | −1.3852 | T=1、D=0.3 时，开启 entropy |
| 11 | 13 | +2.6753 | entropy=true 时，D 从 0.1 增到 0.3 |
| 12 | 14 | −0.4811 | entropy=false 时，D 从 0.1 增到 0.3 |
| 15 | 11 | −2.5562 | D=0.3、entropy=true 时，增加 T |
| 16 | 12 | +1.1603 | D=0.3、entropy=false 时，增加 T |
| 17 | 11 | −3.1710 | D=0.3、entropy=true 时，增加 C |

观察显示存在明显的配置交互：不能概括成“entropy 一定有用”或“蒸馏一定无用”。解释上，teacher entropy 加权会改变蒸馏关注的位置，可能与 token 重建/权重配比产生交互；这只是机制假设，聚合分数没有验证它。

18/19 不在这批固定 run 中，本次未核实其远端完成状态，因此不能声称 causal+distill 的 entropy=false 或 token+causal 组合已被排除。

### 3. 为什么必须选 best，而不能按 final 排名

11 的 AR F1 在 step 3200 达到 69.4611，末步为 66.7520，下降 2.7091 pp；若按 final 排名，会错过当前最佳候选。这个末步值只用于解释峰值回落，不参与主排序。是否是过拟合、优化噪声或评估子集波动，需要完整曲线和重复 seed 才能区分。

best 也有选择偏差：在同一个 validation 子集上多次选 checkpoint，再从多个配置中选赢家，会高估泛化表现。不同阶段有 11/12 个选择点，额外增加可比性限制。

### 4. TF 仅作辅助诊断

06 的 TF best F1 为 69.9576，高于 11 的 69.6548，但其 AR best 为 68.0169，低于 11 的 69.4611（且两者跨阶段）。同阶段更直接的例子是 13 vs 14：13 的 TF best 为 69.6334，高于 14 的 69.0400，但 AR best 为 66.7858，低于 14 的 67.6109。

因此不按 TF 选配置。TF 可用于定位优化/teacher-forcing 差异；各组合总 loss 由不同权重混合，也不宜直接用总 loss 大小评价最终 QA 能力。

## 下一步实验与结论边界

1. **优先复验 10/11/12/16**：QA 基线、当前赢家、entropy 关闭对照、token+distill 候选。统一 batch/global batch、数据版本、代码 commit、评估间隔和训练预算，至少 seeds 42/43/44。每个 seed 按 validation best AR F1 选 checkpoint，再在固定独立 test 集报告 EM/F1/ROUGE-L 的均值与标准差。
2. **验证 entropy 交互**：在 D=0.3 下比较 11/12/15/16 的 2×2（T 开关 × entropy 开关），重复 seed；必要时报告 token 级 teacher entropy 分桶的 KL 与 QA 错误，检验机制假设。
3. **复查 11 的峰值**：确认 step 3200 对应 checkpoint 可用，固定评测集复评；观察邻近 checkpoints 的稳定性。复评聚合分数不能替代独立 test。
4. **统计验证**：保存逐样本预测，按 context 分组 bootstrap 比较候选与 QA-only，避免同一 context 下多个 QA 被当作完全独立样本。仅凭 W&B 聚合历史不能计算可靠的配对显著性。

当前能支持的结论：**11 是这 15 个 run 按 validation best AR F1 排序的首选复验候选；辅助目标效果存在条件依赖。** 不能据此宣布普遍提升、显著性或独立测试性能。

## 数据来源与复核

- [W&B 项目](https://wandb.ai/34520222200995-xiamen-university/MemoryQwen)；逐 run 链接见配置矩阵。
- [原始配置及验证历史快照](data/loss_ablations_20260921.json)：保存本次读取的 config 和含验证指标的全部历史行，可离线复算 best；不包含认证信息。
- [Phase 2 生成脚本](../../scripts/experiments/run_loss_term_ablation_phase2.sh)、[Phase 3 生成脚本](../../scripts/experiments/run_loss_term_ablation_phase3.sh)。实际历史配置以快照为准，不以今天的 baseline YAML 覆盖旧实验。
- [评估与 checkpoint 选择](../../scripts/train.py)、[评估实现](../../src/evaluator.py)、[指标实现](../../src/metrics.py)、[损失实现](../../src/losses.py)。这些是当前 checkout 代码，未据此断言历史 run 的代码 commit 完全一致。
- [早期记录 LOSS_ABLATION.md](LOSS_ABLATION.md) 保留原历史内容；其中 pending 状态和基于末值的判断不代表本次 best-AR 分析。
