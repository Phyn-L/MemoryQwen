# 记忆读取侧（reader）可选项说明

对应历史实施记录：[PLAN_reader_upgrade.md](../history/PLAN_reader_upgrade.md)（计划 + 实测记录），代码在 `src/model.py`、`src/resampler.py`、
`src/losses.py`，配置字段在 `utils/config.py::MemoryConfig`。

**所有字段的默认值都等于旧行为**：一个都不写，或全部保持默认，训练与评估路径和改动前一致（有回归测试钉住）。
只在 `configs/*/train.yaml` 的 `memory:` 段里改值即可，无需碰代码。

---

## 0. 使用范围

以下字段按 `utils/config.py::MemoryConfig` 核对。推荐组合是实验起点，不是普遍优于 OFF 的结论；历史初始 loss 和 CPU 冒烟记录见[实施记录](../history/PLAN_reader_upgrade.md#reader-options-history)。

## 1. 字段总表

| 字段 | 默认 | 作用 | 代价 | 建议 |
|---|---|---|---|---|
| `head_mode` | `linear` | `tied` = 词表头用 backbone 自己的 tied embedding 打分，只训 `D→H` adapter | 省 38.4M 随机参数（1.7B）；每步一次 `V×H×D` 物化（~3 ms） | 作为对照实验选项 |
| `head_init` | `auto` | tied 时 adapter 用 memory projection 初始化（步 0 logits 就有意义） | 无 | 保持 `auto` |
| `init_mode` | `randn` | memory slot 初始化：`token_embed` 用真实 token embedding 行 | 无 | 打开 `token_embed` |
| `init_seed` | `0` | `token_embed` 抽样种子（可复现） | 无 | 随意 |
| `slot_attention` | `isolated` | `isolated`：槽间不可见；`causal`：槽 k 看槽 ≤ k；`bidirectional`：槽间全部互见 | 无新增参数（只改 mask） | 因果/双向需分别消融 |
| `ae_lm_weight` | `0.0` | memory 前缀自编码（冻结 backbone + teacher forcing，500x eq.1） | +1 次 context 长度前向（算力/激活约 +50~100%） | 打开 `1.0` |
| `ae_lm_positions` | `0` | AE 打分的位置数；0 = 全部 | 位置越多越贵（tied 头单次应用 + 256 行分块） | 先 0，显存紧张写 256 |
| `distill_weight` | `0.0` | 把 full-context 分布 KL 蒸馏进 memory 路径 | 每次打分位置一次词表头（teacher 免费） | 打开 `0.2~0.5` |
| `distill_temperature` | `1.0` | 蒸馏温度（`T²·KL`） | 无 | 先 1.0，再试 2.0 |
| `distill_positions` | `256` | 蒸馏打分位置数 | 线性 | 保持 256 |
| `distill_entropy_weight` | `false` | 按 teacher 自身熵加权（把容量放在模型没把握的位置） | 无 | 可试 |
| `distill_topk` | `0` | 把 teacher 截断到 top-k（省 softmax 开销） | 0 = 不截断 | 显存紧时 64 |
| `readout_length` | `0` | 问题期 resampler 输出 R 个位置（全局 memory 仍可缓存） | +R 个位置的前向（R 很小）+2.63M 参数 | 先 8 |
| `readout_layers` | `2` | resampler 层数 | 线性 | 保持 2 |
| `readout_heads` | `4` | resampler 注意力头数 | 无 | 保持 4 |
| `readout_hidden_size` | `256` | resampler 瓶颈宽度（需被 `readout_heads` 整除） | 全宽（2048）会让模块从 2.6M 涨到 67M | 保持 256 |

---

## 2. 推荐的两套配置

### 2.1 稳妥版（先验证 reader 假说，不引入检索侧变化）

```yaml
memory:
  head_mode: tied            # A1
  init_mode: token_embed     # C2
  slot_attention: causal # C2
  ae_lm_weight: 1.0          # B2
  ae_lm_positions: 0
  distill_weight: 0.0        # 先不开蒸馏
  readout_length: 0          # 先不开 resampler
```

配套：`data.max_context_tokens: 512`、`memory_length: 16`、`batch_size: 4~8`。
先在论文的操作点（≤512 token）上确认 `val_teacher_forced/ae_loss` 能降到 2–4 nats、并且 autoregressive F1 抬起来。

### 2.2 全开版（reader + 蒸馏 + 检索侧）

```yaml
memory:
  head_mode: tied
  init_mode: token_embed
  slot_attention: causal
  ae_lm_weight: 1.0
  ae_lm_positions: 0
  distill_weight: 0.3
  distill_temperature: 1.0
  distill_positions: 256
  readout_length: 8
  readout_hidden_size: 256
```

---

## 3. 逐项说明

### A1 `head_mode: tied`

- **做什么**：`context_lm` 的共享词表头不再是随机的 `Linear(D, vocab)`，而是 `W = E @ adapter`
  （`E` 是 backbone 冻结的 tied embedding，只训 `D→H` 的 adapter）。`W` 按参数版本缓存，每个 step 物化一次。
- **为什么**：预训练的词表几何本来就在 `E` 里，随机头要从 256 维瓶颈里重新学"哪个方向是哪个词"。
- **实测**：1.7B 可训练参数 91,740,160 → 53,368,832（Δ 恰为 `151936×256 − 2048×256`）。
- **判据**：memory-only 探针（`reconstruction_loss: context_lm`）的下降斜率是否变陡。若完全不变，
  说明瓶颈在 256 维瓶颈本身，下一步该走 B2（冻结 backbone 当解码器）。
- **坑**：tied 模式的 `state_dict` 与旧 checkpoint 不同（`context_lm_head.adapter.*`）；`linear` 仍是默认且参数名不变。

### C2 `init_mode` / `slot_attention`

- **做什么**：`token_embed` 用真实 embedding 行初始化 M 个槽（`init_seed` 可复现）；`vocab_mean` 用 embedding 均值 + 0.02 噪声。
  `slot_attention` 只控制 encoder pass 的槽间可见性；所有模式均可读取有效 context。
  `isolated` 禁止所有槽间 attention（包括自身）；`causal` 允许槽 k 读取槽 ≤ k；
  `bidirectional` 允许每个槽读取全部槽（包括自身）。context、question、answer 的可见性不变。
  配置仅接受这三个英文值，不兼容旧布尔字段或 `casual` 拼写。旧 checkpoint 内嵌配置需手动迁移后才能加载；参数形状不变。
  `isolated` 与 `causal` 的前部 slots 不依赖后部 slots；`bidirectional` 不保证“完整编码后截断”与“短前缀编码”等价。
- **为什么**：`randn*0.02` 的量级没问题，但方向是随机子空间；槽间可见性让 M 个大 slot 能分工（ICAE 的 memory token 就是普通因果位置）。
- **判据**：前 500 步 memory-only 探针的 EM / nats 是否更快起来；`slot_attention` 的收益在 M=64 时最可能出现。
- **坑**：`token_embed` 抽到低频 token 属实验变量，必要时换 `init_seed`。

### B2 `ae_lm_weight` / `ae_lm_positions`

- **做什么**：新增「冻结 backbone + memory 逐层 KV 前缀 + teacher forcing」的自编码目标
  `P(t_i | memory, t_<i)`，用 backbone 自己的 tied unembedding 打分（零新增参数），
  全 context 位置（或 `ae_lm_positions` 个采样位置）密集监督。
- **为什么**：这是两篇论文的核心机制（500xCompressor eq.1 / ICAE 的 autoencoding + "decoder 就是那个冻结 LLM"）。
  它把"把 latent 变回语言"这件难事交给预训练权重，memory 只需要携带 LM 猜不出来的残差。
- **实测**：见第 0 节表格（2.472 vs 12.140）。
- **判据**：`val_teacher_forced/ae_loss` 进入 2–4 nats 且持续下降；同时看 memory-only 探针与 AR F1 是否改善。
- **坑**：多一次 context 长度前向（算力/激活约 +50~100%），先在 512 token 上迭代；`ae_lm_positions: 0` 在长上下文 + 大 batch 下会很贵。

### B1 `distill_weight` 等

- **做什么**：`T²·KL(p_teacher ‖ p_student)`。teacher = encoder pass 里 context 行（纯因果 LM，**免费**，无额外前向）；
  student = B2 的 memory 前缀 AE pass；两者共用同一个 tied unembedding。位置按 `distill_positions` 采样，
  teacher 侧 `no_grad`，永不接收梯度。
- **为什么**：硬标签里有大量 LM 本来就能猜的内容（ICAE：正常文本 BLEU 99.3 vs 随机文本 0.2），
  分布级目标保留"残差不确定性"，正是 memory 应该承载的部分。
- **实测**：初始 plain KL 2.352 nats；`entropy_weight` 3.500；`topk=64` 2.348；`T=2` 5.907；
  memory token 梯度范数 590（AE 的 CE 版是 28，说明分布级目标梯度更强）。
- **约束**：`distill_weight > 0` 必须 `ae_lm_weight > 0`，否则 `validate()` 直接报错。
- **未做（有意）**：answer 位置的蒸馏需要一次额外 `[context|question|answer]` 因果 teacher 前向（QA 行看不到 context，
  拿不到免费 teacher），步时约 ×2；若要做，建议加 `distill_every_n_steps` 控制频率。

### C3 `readout_length` 等

- **做什么**：保留可缓存的全局 memory，另加一个**问题期** resampler：R 个可学习 latent 对
  `cat([question(masked), memory])` 做 cross-attn，输出 `[B, R, H]` 作为输入 embedding 插到 question 与 answer 之间。
  question/memory 先降到 `readout_hidden_size` 瓶颈（默认 256，与 per-layer decoder 同宽）。
- **为什么**：2048 token 里与某道题相关的往往只占 1%，而全局 memory 编码时不知道要问什么，只能"什么都留"。
- **实测**：1.7B 下 R=8 时 resampler 2,633,728 参数；`qa_loss` 反向后 39/39 参数张量拿到非零梯度；
  生成路径（`generate_answers_with_prefix`）会调用它。输出投影按 token-embedding 量级初始化，步 0 是无 readout 路径的小扰动。
- **判据**：AR F1 增量（应最直接）；同时按 context 划分 held-out 问题，区分"装了内容"与"记住这道题"。
- **坑**：`readout_length` 改变了每行序列长度（logits/labels 变成 `Q+R+A`，R 段标签为 `-100`），
  既有评测脚本无需改（evaluator 走同一批函数），但自己解析 logits 时要注意。

---

## 4. `validate()` 会拦住的组合

- `head_mode ∈ {linear, tied}`；`head_init ∈ {auto, random, memory_projection}`
- `init_mode ∈ {randn, token_embed, vocab_mean}`
- `ae_lm_weight ≥ 0`、`ae_lm_positions ≥ 0`（0 = 全位置）
- `distill_weight ≥ 0`、`distill_temperature > 0`、`distill_positions ≥ 0`、`distill_topk ≥ 0`
- **`distill_weight > 0` 要求 `ae_lm_weight > 0`**
- `readout_length ≥ 0`；`>0` 时 `readout_layers > 0`、`readout_heads > 0`、`readout_hidden_size > 0` 且被 `readout_heads` 整除

---

## 5. 诊断口径（比开关本身更重要）

1. **保留 `reconstruction_loss: context_lm` 当探针**：它是唯一 memory-only 的诚实指标。想省算力就把
   `context_lm_positions` 降到 64，而不是关掉。
2. **不要用 BLEU/ROUGE 当主判据**：teacher forcing 条件下大部分"复述得像"来自 LM 先验（ICAE 的随机文本实验：99.3 → 3.5 → 0.2）。
   用 `first_token_em`、逐位置 EM 曲线，以及**空 memory 基线**（把 memory 置零/置常数的对照）。
3. **按 context 划分 held-out 问题**：SQuAD 的 16498 个 QA 行只覆盖 10531 个唯一 context，可以按 context 切分，
   用来判断 memory 是否把问题无关的内容真的装进去了。
4. **wandb 面板**：`val_teacher_forced/{loss, qa_loss, ppl, reconstruction_loss}` 看目标，
   `val_autoregressive/{f1, em, rouge_l, first_token_em}` 看下游；训练侧的 `ae_loss` / `distill_loss`
   只在 `train/*` 里（验证不跑它们），两个 section 由 `scripts/train.py::eval_log_payloads` 决定。

---

## 6. 已删除的选项

- `contrastive_weight` / `contrastive_temperature` / `contrastive_margin`（`memory_contrastive_loss`）**已删除**：
  三个 config 与全部历史 run 都是 `0.0`，没有测试，且 [IMPROVEMENTS](../history/IMPROVEMENTS.md) E8 已审计出它的 positive 项
  `(z·z)/T ≡ 1/T` 是常数（退化成纯阈值排斥项）。若日后要做"同 context 为正、跨 context 为负"的对比目标，
  请新加一个带测试的实现，不要复活旧实现。

---

## 7. 回退

把 `head_mode` 改回 `linear`、`init_mode` 改回 `randn`、`slot_attention` 改回 `isolated`、
`ae_lm_weight`/`distill_weight`/`readout_length` 全部改回 `0`，即完全回到升级前的数值路径
（`tests/` 里每个开关都有"关闭即等价"的回归测试）。

---

## 8. 怎么验证这些开关有用

见 [AB_H200.md](../experiments/AB_H200.md)：`configs/qwen-1.7b/ab_h200_{on,off}.yaml` 两臂除 6 个开关外逐字段
相同（`tests/test_ab_configs.py` 断言），步数以所选配置、进程数和实际数据量打印的 schedule 为准，用 `scripts/archive/run_ab.sh` 顺序跑完两臂。
之前的 `0rj6x1xc`（ctx2048/M64/1ep）与 `vry7n1sw`（ctx512/M16/3ep）形状不同，只能当规模参照，
不能用来判断开关的好坏。
