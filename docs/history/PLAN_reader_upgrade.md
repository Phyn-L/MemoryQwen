# 记忆读取侧升级计划（A1 / B1 / B2 / C2 / C3）

> 历史快照：2026-09-18 归档。正文状态、数值、路径和预计完成时间属于当时记录，本次未重新运行实验或核实远端状态。当前操作见[运行指南](../guides/RUNNING.md)，实验状态见[实验索引](../experiments/README.md)。

初始计划状态：**当时待执行**（commit 1）。后文包含实施与验证记录，请按各节时间阅读；当前字段见[reader 选项](../guides/READER_OPTIONS.md)。
机器：4x4090（`/data/lz/MemoryQwen`，HEAD `9d62f53`）
参考：ICAE（ICLR 2024）、500xCompressor（arXiv:2408.03094），逐页分析见 `papers/ICAE_report.md`、`papers/500xCompressor_report.md`

---

## 0. 为什么改这里（诊断）

已有的实测锚点：

| 配置 | 压缩比 | AR F1 | 相对 full-context(0.6847) 保留 |
|---|---|---|---|
| full-context bypass（ICL 基线 0.6810） | 1× | **0.6847** | 100% |
| M=8 | 256× | 0.2534 | 37% |
| M=64 | 32× | 0.3636 | **53%** |
| 500xCompressor 500→16 | 31× | — | ~73%（抽取式 QA 平均 F1） |
| ICAE 512→128 | 4× | — | ~80%（k=256 才追平） |

两篇论文在**同量级压缩比**下都比我们高约 20 个相对百分点，而它们的共同点是：

1. **reader 就是那个冻结的预训练 LLM**（ICAE 原文 "the untouched target LLM as the decoder"；500x 解码器零新增参数），memory 只负责携带"LLM 猜不出来的残差"，且重建是 **teacher-forced 条件**的 `P(t_i | memory, t_<i)`；
2. memory 以**逐层 KV**（500x）或**序列前缀**（ICAE）注入冻结模型，走的是模型自己的注意力与词表头。

我们现在的 reader 是：`MemoryDecoder`（每层 256 维瓶颈，只 attend memory）× 随机初始化的 `context_lm_head: 256→151936`，目标是在 **memory-only** 条件下复述全部 context token（采样 256 个位置）。即：**memory 侧的信息量不比论文少（28 层 × 2048/token ≈ 500x 的逐层 KV），但读出口要从零学一个语言模型头**。因此本轮全部改动集中在**读取侧与目标函数**，不改 writer 的骨架。

---

## 1. 总原则

- **P1｜默认关闭**：所有新行为由新增配置字段控制，默认值 = 现状。关闭时数值路径必须与改动前一致，并有测试钉住（不是"看起来一样"）。
- **P2｜逐项提交**：每项一个 commit；commit 前必须 `pytest tests/ -q` 全绿 + 跑一次短程冒烟。
- **P3｜不碰用户未提交改动**：`configs/4090/qwen-1.7b/baseline/train_baseline.yaml`、`configs/4090/qwen-8b/baseline/train_baseline.yaml` 在工作区是未暂存状态，全程保持不动；只 `git add <显式路径>`，禁止 `git add -A`。
- **P4｜状态字典兼容**：A1 的 `linear` 模式参数名必须仍是 `context_lm_head.weight`，旧 checkpoint 可直接加载。
- **P5｜每项都要有等价性/形状/梯度三类测试**，等价性测试用**微缩 Qwen3**（`Qwen3Config` 随机初始化，无下载）或 `_FakeQwen`。

---

## 2. 环境与改动前基线

- 仓库：`/data/lz/MemoryQwen`（`origin git@github.com:Phyn-L/MemoryQwen.git`），HEAD `9d62f53`。
- Python：`/home/lz/miniconda3/envs/shine/bin/python`（3.12.0，torch 2.5.1+cu124，transformers 5.16.1，datasets 4.4.1，accelerate 1.14.0，pytest 9.1.1，**4 卡可见**）。注意 `/data/lz/miniconda3/envs/shine` 是另一个不完整的 env，别用。
- **改动前基线记录**：`pytest tests/ -q` → **46 passed in 7.27s**。
- 编辑流程：本地镜像改 → `rsync` 回服务器 → 服务器跑测试/冒烟 → 服务器 `git commit`（本地不是 git 仓库，服务器是唯一事实来源）。

---

## 3. 改动清单

### A1｜词表头继承预训练 tied embedding 几何

**改什么**：`src/model.py` 新增 `VocabularyHead`，替换裸 `nn.Linear(256, 151936)`；`src/losses.py` 不再假设 `head.weight` 存在；`utils/config.py` 增字段。

**为什么**：现在 38.9M 随机参数（fp32 下 156 MB 参数 + ~470 MB AdamW 状态/卡）承担"把 256 维瓶颈映射到词表"的全部工作，且 token 几何要从零学；Qwen3-1.7B `tie_word_embeddings: true`，输入 embedding 就是 lm_head，几何本来就存在。

**怎么改**：

```python
class VocabularyHead(nn.Module):
    """context_lm 的共享 unembedding。

    mode="linear"（默认，行为与以前逐位相同）:  Parameter weight [V, D]
    mode="tied"  :  W = E @ adapter.weight，E = 冻结的输入 embedding [V, H]，
                    只有 adapter [H, D]（0.52M）可训练；W 每次 forward 物化一次并缓存。
    """
    compute_dtype   # losses 用它做 chunk 的 dtype（替代 head.weight.dtype）
    def refresh(self)            # 丢掉物化缓存（每次 forward 开头调用）
    def materialized_weight(self)  # linear: weight；tied: E @ adapter.weight
    def forward(self, hidden):     # F.linear(hidden, self.materialized_weight())
```

- **必须物化**：`(h @ A.T) @ E.T` 每行 312M MACs，物化后每行 38.9M（与现在相同）；物化成本 `151936×2048×256 ≈ 1.6e11 FLOPs ≈ 3 ms/卡`，盈亏平衡点仅 292 行，而每步有 8.6 万行。
- **初始化**（`head_init="memory_projection"`，tied 模式默认）：`A = decoders[0].memory_projection.weight.T`（[H,D]），此时 `logits = h @ W.T = (W_mem h) @ E.T`，即"预测与 memory 隐状态最对齐的 token"，比随机方向好得多。
- 缓存生命周期：`MetaLoRA.forward()` 开头调 `self.context_lm_head.refresh()`；`torch.utils.checkpoint` 的反向重算会复用同一份物化结果（梯度仍正确）。
- 配置：`memory.head_mode: "linear" | "tied"`（默认 `"linear"`）、`memory.head_init: "auto" | "random" | "memory_projection"`（默认 `"auto"`：tied → `memory_projection`，linear → `random`）。`validate()` 校验取值。
- 实现细节（改动时踩到并修掉的两个坑，均有测试钉住）：物化权重必须**提到 `torch.utils.checkpoint` 之外作为输入传入**，否则重算时缓存命中会让"保存的张量数"不同（`CheckpointError`）；`E` 是 bf16 而 adapter 是 fp32，相乘要先把 `E` 转到 adapter 的 dtype（保持 fp32 精度，代价是一次性的 ~1.2 GB 转置拷贝，分配器会复用）。
- `is_trainable_parameter_name` 已含前缀 `context_lm_head.`，tied 模式的参数名是 `context_lm_head.adapter.weight`，**无需改规则**（并有测试断言这一点）。

**怎么验证**：
1. `tests/test_vocab_head.py`：
   - linear 模式：`forward` == `F.linear`；参数名恰为 `context_lm_head.weight`；trainable 参数数不变（回归钉）。
   - tied 模式：`materialized_weight() ≈ E @ adapter.weight`；logits 与手算一致；`adapter.weight.grad` 非空且 `E.grad is None`；可训练参数 = D×H。
   - `refresh()` 语义：改 adapter 权重后，未 refresh 结果不变、refresh 后改变。
   - 初始化：tied+memory_projection 时 `W ≈ E @ W_mem.T`。
   - `context_lm_loss` 在 tied 头下可前向/反向，loss 有限。
   - 默认值检查：不配置时构造出的模型与改动前 `state_dict().keys()` 完全一致。
2. 全套 `pytest` 仍 46+ 通过（`tests/test_eval_logging.py` 会间接覆盖 loss 路径）。
3. 冒烟：M=16、512 token 短跑 200 步，观察 memory-only nats/token 下降斜率（当前 11.94→10.95/30 步，几乎贴着 ln vocab=11.93）。

**风险/回退**：tied 模式 state_dict 与旧 checkpoint 不兼容（仅当选，不影响默认）；物化缓存若忘记 refresh 会用到上一步的 W —— 由 `forward()` 内 refresh + 测试钉住。回退 = `head_mode: linear`。

---

### C2｜memory token 初始化 + 放开 memory↔memory 因果注意力

**改什么**：`src/model.py::MetaLoRA.__init__` 末尾按配置初始化 `memory_tokens`；`build_block_causal_mask` 增加 `slot_attention` 参数；`utils/config.py` 增字段。

**为什么**：
1. `randn(M,H)*0.02` 的**量级**与 Qwen embedding init 相当，但**方向**是随机子空间，不在真实 token embedding 的流形上；用真实 embedding 初始化是零风险改进。
2. 现有 mask 显式禁止 memory 之间互相看（源码注释 `no memory <-> memory`）。但 ICAE 的 k 个 memory token 是普通因果序列、彼此可见，可以协调分工；M=64 时这一点最可能重要。放开它的显存/算力代价为 0。

**怎么改**：
- `memory.init_mode: "randn" | "token_embed" | "vocab_mean"`（默认 `"randn"`，逐位保持现状）+ `memory.init_seed: int = 0`。
  - `token_embed`：用固定种子从 `[0, vocab)` 无放回抽 M 个 id，`memory_tokens.data.copy_(E[ids])`（转 trainable_dtype）。
  - `vocab_mean`：`E.mean(0) + 0.02*randn`。
  - 执行位置：`__init__` 末尾（`set_trainable_dtype` 之后），避免被 `self.to(dtype=qwen_dtype)` 冲掉。
- `memory.slot_attention: str = "isolated"`：致 `build_block_causal_mask(..., slot_attention=...)`，为 `allowed[:, m0:q0, m0:q0]` 填 causal tril（memory token 无 padding，恒为真实位）。
- `build_continuation_mask` **不需要改**（它的行只有 question/answer，memory 是外部 KV），所以既有的"空 context 等价"测试继续成立。

**怎么验证**：
- `tests/test_masks.py` 增：`slot_attention="isolated"` 时新实现与旧实现**逐位相等**（把旧行为写成断言矩阵）；`causal` 时 memory 行 i 只见 memory j≤i，且 question/answer 行与 isolated 时逐位相同。
- 新增 `tests/test_memory_init.py`：三种 init 的形状/dtype；`randn` 默认与旧实现（固定种子）逐位相等；`token_embed` 的值确实来自 `E`（逐行 `isin` 断言）；`vocab_mean` 的均值接近 `E.mean(0)`。
- 全套 pytest。
- 冒烟：M=64 下 `slot_attention` causal/isolated各跑 200 步，比较 loss 与 trainable 参数量（应完全相同）。

**风险/回退**：`token_embed` 取到低频/特殊 token 可能不适配，属实验项；默认关闭即回退。

---

### B2｜融合前向（省一遍前向 + 免费 teacher logits）+ KV 前缀版 AE（= 500x eq.1）

**改什么**：`src/model.py` 新增 `fused_forward` 与 `autoencode_with_memory`；`src/losses.py` 新增 `sequence_lm_loss`；`utils/config.py` 增字段；`scripts/train.py` 接线与日志键。

**为什么**：这是两篇论文的核心机制，也是"同一压缩比下差 20 个点"的最可能原因。现有 `context_lm` 要求 memory-only 复述全文；改成"冻结 Qwen + memory KV 前缀 + teacher-forced 条件"后，解码由预训练权重完成（`P(t_i | memory, t_<i)`），memory 只需携带残差。

**怎么改**：

1. **KV 前缀版 AE**（`autoencode_with_memory`，主收益项）：
   ```python
   mask = build_continuation_mask(context_mask, empty, M, dtype)   # context 行看全部 memory + 因果自身
   positions = (L + M) + arange(L)                                  # memory KV 仍在 [L, L+M)
   out = self.qwen(inputs_embeds=context_embeds, attention_mask=mask,
                   position_ids=positions, past_key_values=prefix.memory_cache,
                   use_cache=False, return_dict=True)
   return out.last_hidden_state            # 交给 sequence_lm_loss 用词表头打分
   ```
   复用 `build_continuation_mask` 而不是新写 mask：它的 question 行规则（看全部 memory + 因果自身）正是 AE 需要的 teacher-forced 结构。**位置约定与现有 QA 路径一致**（memory 在原位置，解码序列排在其后），不需要对 RoPE 做任何"重新旋转"。
2. **`sequence_lm_loss(hidden, labels, head, mask, positions=None, max_logits_rows=...)`**：单次词表头应用（不是逐层 28 次）+ 分块 + `torch.utils.checkpoint`；`positions` 支持全量或采样。成本：B×L 行 × 38.9M MACs（B=4, L=2048 → ~0.6 TFLOPs，约 1% 步开销）。
3. **teacher logits 免费拿（原"融合前向"）**：实现时确认了一个约束 —— 训练 collate 会把 QA 行**展平**（`qa_context_indices` 把 B×qa 行映射到 B 个 context，见 `src/data.py:220-223`），所以"把 QA 并进 encoder 一次前向"在本仓库布局下省不掉前向：两个 pass 处理的是不同的 token 集合（B×(L+M) 与 B×qa×(Q+A)），总 token 数不变。融合的真实价值是**免费 teacher logits**，而这一点可以直接从既有 encoder pass 拿到：`build_block_causal_mask` 的 context 行只 attend 更早的 context，因此它们的末层隐状态**就是**纯因果 LM 分布。于是给 `ContextPrefix` 增加 `context_hidden`（复用已经算出来的 `hidden_states`，零额外显存/算力），B1 的 teacher 由此得来。
   （若日后仍要真正的单次前向，需要新增"每个 context 一行、QA 块之间互相隔离"的块对角 mask 并改 collate；评估路径与生成路径无法复用该布局，收益仅剩 kernel 启动开销，故本轮不做。）
4. 配置：`memory.ae_lm_weight: float = 0.0`（0 = 关闭）、`memory.ae_lm_positions: int = 0`（0 = 全部 context 位置，>0 时按 `sample_positions` 采样）。
5. 日志：`scripts/train.py::LOSS_KEYS` 增加 `"ae_loss"`（缺失时不记录，兼容旧的 eval payload 测试）。
6. AE 用 **backbone 自己的 tied unembedding**（新增零参数 `TiedUnembedding`）打分，不新学分类头；memory-only 的 `context_lm` 探针继续用 `context_lm_head`，两个目标互不污染。

**怎么验证**：
- `tests/test_ae_lm.py`（微缩 Qwen3，CPU）：
  1. **融合 vs 两段式数值等价**：同一模型/输入，`fuse_qa_pass=false` 与 `true` 的 QA logits `allclose(atol=1e-4)`（固定种子、`sdpa`/eager 一致）；
  2. **mask 语义**：融合布局里 QA 行看不到 context（断言 `allowed` 矩阵），memory 行只看 context；
  3. AE pass 的 hidden 形状 = `[B, L, H]`，labels = `context_ids`；`ae_lm_weight=0` 时 `forward` 不产生 AE 相关计算（回归：与改动前 `state_dict`/logits 一致）；
  4. AE 梯度流向：`memory_tokens.grad` 非空、冻结的 backbone 权重 `grad is None`；
  5. `sequence_lm_loss` 在 `positions` 采样与全量两种模式下都有限且可分块（chunk 边界与整块等价）。
- 全套 pytest。
- 冒烟（关键判据）：512 token、M=16，`ae_lm_weight=1.0`，看 **AE nats/token 是否进入 2–4**（对照：现在 memory-only 是 ~11.9），以及 memory-only 探针（保留的 `context_lm`）与 AR F1 是否同步改善。
- **已完成的实测（真实 1.7B、真实英文文本、memory 尚未训练，CPU 冒烟）**：

  | 目标 | nats/token |
  |---|---|
  | teacher：纯因果 LM（= full-context bypass） | **2.103** |
  | **ae：memory 前缀 + teacher forcing（500x eq.1）** | **2.472** |
  | probe：memory-only（旧目标） | **12.140**（ln vocab = 11.931） |

  说明冻结 LM 解码把目标一开局就放回语言模型量级（与 teacher 差 0.37 nats，这 0.37 正是 memory 要学的部分），而旧目标贴着随机分类基线。`memory_tokens` 梯度范数 28.4，冻结参数梯度为 0。

**风险/回退**：融合路径改动了 `forward` 的默认结构 → 由 `fuse_qa_pass=false` 保持旧路径，且等价性测试钉住；AE pass 多一次上下文前向（算力/激活约 +1 倍），冒烟用小上下文。回退 = 权重置 0 / 开关置 false。

---

### B1｜KL 蒸馏（免费 teacher 的 context 位置 + 可选 answer 位置）

**改什么**：`src/losses.py` 新增 `kl_distill_loss`；`src/model.py` 暴露 teacher/student logits；`utils/config.py` 增字段；`scripts/train.py` 接线。

**为什么**：复述硬标签里大部分是 LM 先验（ICAE：正常文本 BLEU 99.3 vs 完全随机 0.2），让 memory 去背这些是浪费；而"逐字命中答案"的任务更需要**分布级**对齐。最贴下游的形式是：teacher = 看到完整 context 的分布（就是 0.6847 的 bypass 路径），student = memory + question —— 即把 full-context 能力蒸馏进 memory 路径。

**怎么改（已实现）**：
- `src/losses.py::kl_distill_loss(student_hidden, teacher_hidden, head, mask, temperature, positions, max_logits_rows, topk, entropy_weight)` → `T² · KL(p_tea ‖ p_stu)`（forward KL，覆盖式），按位置取均值；teacher 侧 `no_grad`，永不接收梯度。
- **teacher 来源 = encoder pass 的 context 行**（B2 顺带拿到的 `ae_teacher_hidden`，零额外前向）：context 行只 attend 更早的 context，本身就是纯因果 LM，也就是 full-context bypass 分布。
- 位置选择：`sample_positions`（均匀采样）；`entropy_weight=True` 时按 teacher 自身熵加权（免费，不需要额外前向）；`topk>0` 时截断 teacher 到 top-k。
- 配置：`memory.distill_weight`（默认 0）、`distill_temperature`、`distill_positions`（默认 256）、`distill_entropy_weight`、`distill_topk`；`validate()` 要求 `distill_weight > 0` 时必须 `ae_lm_weight > 0`。
- **未实现（有意留作后续）**：answer 位置的蒸馏需要一次额外的 `[context | question | answer]` 全因果 teacher 前向（`build_block_causal_mask` 禁止 QA 看 context，所以拿不到免费 teacher），步时约 ×2。当前只做 context 位置版本；若日后要做，先加 `distill_every_n_steps` 控制频率。

**怎么验证**：
- `tests/test_distill.py`：student==teacher → loss ≈ 0；`student_logits` 需要梯度时 `grad` 非空、`teacher` 不建图；`T` 缩放（T=2 与手算一致）；surprisal 采样在构造的分布上选出预期位置；`weight=0` 时 `forward` 与改动前逐位一致。
- 冒烟：512 token、M=16，`distill_weight` ∈ {0, 0.3} 对照，看 AR F1（squad 子集）与 memory-only 复述指标。
- **已完成的实测（真实 1.7B、真实英文文本、memory 尚未训练，CPU 冒烟）**：plain KL = 2.352 nats，`entropy_weight` = 3.500，`topk=64` = 2.348，`T=2` = 5.907；`memory_tokens` 梯度范数 590（远大于 AE CE 的 28，说明分布级目标的梯度更强），冻结参数梯度为 0。

**风险/回退**：KL 与 CE 的权重平衡（`qa_weight`/`reconstruction_weight` 已存在）；teacher/student 词表不一致会直接报错（同模型，无此风险）。回退 = `distill_weight: 0`。

---

### C3｜可缓存全局 memory + 问题期 resampler

**改什么**：新增 `src/resampler.py::QuestionResampler`；`build_block_causal_mask` / `build_continuation_mask` 增加 `readout_length`；`forward_qa_with_prefix`、`fused_forward`、`generate_answers_with_prefix` 在 question 与 answer 之间插入 readout 块；`utils/config.py` 增字段。

**为什么**：2048 token 里与某道题相关的常常只占 1%，而当前 memory 是**问题无关**的（编码时不知道要问什么）→ 必须"什么都留"，这正是 45% 差距的来源。变体 2 保留"一次压缩、多次查询"的缓存优势，只在问题期加一次**极廉价**的再采样（K/V 只有 M 个 key，代价可忽略）。

**怎么改（已实现）**：
- `src/resampler.py::QuestionResampler`：R 个可学习 latent 对 `cat([question_embeds(masked), memory], dim=1)` 做 `readout_layers` 层 cross-attn+FFN，输出 `[B, R, H]`，**作为输入 embedding 插入序列**（由 backbone 自己算 K/V，因此不需要 per-layer prefix 投影）。question 与 memory 先投影到 `readout_hidden_size`（默认 256，与 per-layer decoder 同宽度）的瓶颈再进 cross-attn —— 全宽版本在 H=2048 时每层 ~34M 参数，瓶颈版整模块 ~2.6M。
- **输出投影按 token-embedding 量级初始化**（`std = 0.02/sqrt(width)`）：read-out 是插进输入流的，步 0 时应当是"无 read-out 路径的小扰动"，而不是一个 out-of-distribution 的大激活。测试里钉住了这一点（初始 `out.std() < 0.05`）。
- mask：布局扩展为 `[context | memory | question | readout | answer]`（块掩码）与 `[memory(prefix) | question | readout | answer]`（续写掩码）；readout 行看"全部 memory + 全部有效 question"，answer 行看"memory + question + readout + 因果 answer"，**question 行看不到 readout**；`readout_length=0` 时逐位等于旧实现（回归钉）。
- 训练/评测：`forward_qa_with_prefix` 序列变为 `[question, readout, answer]`，readout 段标签填 `-100`（`qa_loss` 的 shift 语义不变，answer 标签对齐由测试钉住）；`generate_answers_with_prefix` 的 prefill 变为 `[question, readout]`，readout 位置取"每行真实 question 末尾之后"（`start + valid + r`），增量步位置为 `start + valid + R + step`，cache 前缀 mask 同步包含 readout。`evaluator` 的两条路径都走这些函数，因此自动生效。
- 配置：`memory.readout_length`（默认 0 = 关闭）、`readout_layers`（2）、`readout_heads`（4）、`readout_hidden_size`（256）；`validate()` 检查可整除。

**怎么验证**：
- `tests/test_readout.py`（微缩 Qwen3 + `_FakeQwen` 双路径）：
  1. `readout_length=0` → 前向/生成与改动前逐位一致；
  2. 开启时 mask 逐位断言（readout 看不到 answer；answer 看得到 readout；question 看不到 readout）；
  3. 生成路径：tiny 模型上"带 readout 的逐步生成"与"一次性构造同样前缀的前向"argmax 一致（防止位置/cache 前缀 mask 出错）；
  4. 参数量/形状断言；`evaluator` 两条路径在 tiny 模型上可跑通。
- 全套 pytest。
- 冒烟：M=16、`readout_length` ∈ {0, 8} 对照，看 AR F1 与 held-out 问题子集（按 context 划分）上的差异。
- **已完成的实测（真实 1.7B、CPU 冒烟）**：`readout_length=8` 时 resampler 2,633,728 参数（全部可训练），可训练参数总量 80,152,576；前向 `[question|readout|answer]` 的 logits/labels 形状 = Q+R+A、readout 行标签为 -100、answer 标签对齐保持；`qa_loss` 反向后 **39/39 个 resampler 参数张量拿到非零梯度**（说明 answer 行确实 attend 到 readout），冻结参数无梯度；`generate_answers_with_prefix` 正常产出且调用 resampler。

**风险/回退**：生成路径最易错（位置/bookkeeping）→ 三重护栏：关闭即等价、mask 逐位断言、tiny 模型 greedy 一致性。回退 = `readout_length: 0`。

---

## 4. 依赖与执行顺序

```
A1 (tied head)  ──┬─► B2 (fused + KV-prefix AE) ──► B1 (KL 蒸馏，teacher/student 依赖 B2 的两条路径)
C2 (init/mask)  ──┘
C3 (resampler)  ← 独立，但建议放在 reader 修好之后（否则分不清是 reader 弱还是 memory 无问题条件）
```

计划提交序列：`docs(plan)` → `A1` → `C2` → `B2` → `B1` → `C3`（可选：`chore(smoke)` 冒烟脚本收尾）。

**实际提交（全部落在 4x4090 的 `/data/lz/MemoryQwen`，基线 `9d62f53`，用户未提交的 config 改动全程保持未暂存）**：

| commit | 内容 | 测试数 |
|---|---|---|
| `0074080` | docs(plan) 本文件 | 46 |
| `4565919` | A1 tied 词表头（VocabularyHead + config + 两个测试文件） | 65 |
| `5ccfbda` | C2 slot 初始化 + memory↔memory 因果注意力 | 75 |
| `b00add4` | B2 memory 前缀 AE（=500x eq.1）+ 免费 teacher logits | 87 |
| `8d1aa1d` | B1 KL 蒸馏（context 位置，teacher 来自 encoder pass 的 context 行） | 97 |
| `7286c85` | C3 问题期 resampler（256 维瓶颈，2.63M 参数） | 108 |
| `7558cae` | logging：训练侧 ae_loss/distill_loss 进 teacher-forced 损失面板 | 109 |

**实现完成后新增/修正的两处计划外内容**：
1. `docs/history/PLAN_reader_upgrade.md` B2 节记录的"融合前向"约束（collate 展平 QA 行 → 单次前向省不掉，改为免费拿 teacher logits）。
2. `7558cae`：B1/B2 的损失只在训练时计算，evaluator 不产生它们，所以仅改 `LOSS_KEYS` 永远不会被记录（端到端冒烟实测发现）；`eval_log_payloads` 增加可选 `train_metrics` 后，wandb 里可见 `val_teacher_forced/ae_loss` 与 `val_teacher_forced/distill_loss`。

**仍未完成（阻塞于资源）**：GPU 上的端到端训练冒烟（真数据、真 1.7B、小 batch、约 30 步）。4x4090 四张卡在实现期间被其他任务占满（每卡 ~20/24 GB、利用率 40–100%），未打扰；已用"真实 1.7B + 真实文本的 CPU 冒烟"和"tiny Qwen3 + 完整 `scripts/train.py` 的端到端 CPU 冒烟（五项功能全开、TF+AR 评测各一次）"代替。等有空闲卡时补跑。

## 5. 端到端冒烟方案（每项之后都跑）

- 只用 1 张空闲卡：`CUDA_VISIBLE_DEVICES=0 NUM_PROCESSES=1`（先用 `nvidia-smi` 确认该卡空闲，避免干扰他人任务）。
- 环境：`MODEL_ROOT=/data/lz/hf_cache/hub DATA_ROOT=/data/lz/contexts/aggregated WANDB_MODE=offline`（4090 无 `scripts/env.local.sh`，这些正好是默认值，只是显式声明）。
- 小样本配置（临时文件，不入库）：`train_max_samples: 64`、`validation_max_samples: 8`、`batch_size: 2`、`epochs: 1`、`teacher_forced_every: 50`、`autoregressive_every: 1000000`、`max_context_tokens: 512`、`memory_length: 16`，跑 ~30 步。
- 判据：
  1. **全关**时首 10 步 loss 与改动前基线一致（|Δ| < 1e-3）；
  2. 逐项开启时 loss 有限（无 NaN）、`trainable params` 增量与预期一致（A1 −38.4M；C3 +R 相关；其余 0）；
  3. `torch.cuda.max_memory_allocated()` 增量符合预期（AE 段约 ×2 激活，先只在 512 token 上验证）。

## 6. 明确不做（避免重复论文里不存在的机制）

- 压缩比 curriculum、compressed-token 初始化技巧、层选择消融 —— 两篇论文都没有；
- 2048 token + M=1 的极端设置 —— 超出论文验证范围（480× 上限、96–480 token 上下文）；
- 用 BLEU/ROUGE 作为主判据 —— 会被 LM 先验刷高（ICAE：BLEU 0.98 / EM 0.6），改用 `first_token_em`、逐位置 EM、空 memory 基线。

## 原 README 中的扩展想法及后续状态

来源：整理前 README 第 800–825 行。下面的 “not implemented” 是旧表述：当前已有可选 AE-LM 和 tied head；不能据此认定相关能力仍未实现。grouped decoder sharing 仍作为未实施设想保留。

#### Alternatives not implemented

- **Let Qwen itself be the reconstruction decoder** (AutoCompressor style): run the
  backbone over `[memory, context[:-1]]` with a causal mask that lets context positions
  attend to the memory, and compute the cross entropy with the frozen `lm_head`. The
  memory is then trained by the same machinery that consumes it, and the per-layer
  decoders can be deleted. More faithful, but it costs roughly another full prefix-length
  forward pass per step.
- **Keep the `H`-width output and reuse Qwen's frozen `lm_head`.** This keeps the
  pretrained unembedding geometry instead of learning a head from scratch, at eight times
  the head cost (see above). Useful if the learned head turns out to be the bottleneck.


## Possible decoder extension: learned layer embeddings with grouped sharing

A resource-efficient follow-up is to share one bottleneck decoder within each
group of adjacent Qwen layers. Before decoding, add a learned layer embedding to
the projected memory tokens and/or positional queries so that the shared decoder
can condition its reconstruction on the source layer. For example, 28 Qwen layers
can be divided into seven groups of four layers, reducing decoder parameters by
approximately 4x while retaining layer identity. This is an exploratory model
variant rather than the current implementation and should be compared against
independent decoders with matched bottleneck width. Useful ablations include group
sizes 1, 2, 4, and 28, with and without learned layer embeddings, evaluated using
per-layer reconstruction loss, QA metrics, peak memory, and training throughput.


<a id="reader-options-history"></a>

## 从 reader 选项迁入的历史测量

以下为原 reader 说明的初始测量和 CPU 冒烟记录，未在本次重测。

## 0. 一句话总览：这些开关在修什么

旧路径的瓶颈不在 memory 侧，而在**读出口**：memory 的逐层隐状态（28 层 × 2048/token）被喂给一个
**从零学的 256 维瓶颈解码器 + 随机初始化的 `256→151936` 词表头**，并且要求在 **memory-only**（看不到任何前文）
的条件下复述整个 context。实测（真实 1.7B、真实英文文本、memory 尚未训练）：

| 目标 | nats/token |
|---|---|
| teacher：纯因果 LM（full-context bypass） | 2.103 |
| **AE：memory 前缀 + teacher forcing（B2 开启后）** | **2.472** |
| probe：memory-only（旧目标，默认仍在跑） | 12.140（≈ ln vocab = 11.931） |

也就是说：把解码器换成冻结的 backbone（B2）之后，目标函数一开局就回到语言模型量级（与 teacher 只差 0.37 nats，
这 0.37 就是 memory 要学的东西）。下面的开关就是围绕这一点组织的。

---


已在 CPU 上用真实 1.7B（真 tokenizer）与 tiny Qwen3 的完整 `scripts/train.py` 跑通（含 TF/AR 评测）：
`val_teacher_forced/{ae_loss, distill_loss, loss, qa_loss}` 都会出现在 wandb 的 teacher-forced 面板里。
