# MemoryQwen 现状诊断与改进方向

> 证据来源：`wandb/run-*`（离线 `.wandb` LevelDB 记录，已逐条解析）、`outputs/icl_baseline/`、
> 以及本次新增的 GPU 复现实验（脚本见 `.tmp_analysis/`）。
> 所有实验都在 SQuAD v1.1 dev（`/data/lz/contexts/aggregated/squad/validation.jsonl`
> 前 300 个 context 的首个 QA）上做，checkpoint 为 `outputs/Qwen1.7B_20260916_024918/step-20000.pt`。

---

## 0. 结论摘要

1. **"0.25 : 0.745 ≈ 1/3" 这个对比本身不成立**，口径至少差 6.4 F1 分：
   用项目自己的 `src/metrics.py` 去评 ICL baseline，0.7451 会掉到 **0.6810**。
2. **"自回归 F1 = 0.0044" 是过期代码路径的历史数字**。用当前代码 + 当前 checkpoint 重测，
   自回归 official F1 = **0.3784**（EM 0.2567），不是 0.0044。该数字来自 commit `7e2099e`
   （context 聚合 / prefix-KV 重构之前的旧 forward + 无 KV cache 的旧 generate）。
3. **真正的主要瓶颈是 8 个"问题无关"的 memory token**：把 memory 旁路掉、让
   question/answer 直接看完整 context KV，同一个 LoRA 模型能拿到 official F1 **0.6847**
   （≈ 与 ICL baseline 0.6810 持平）。也就是说 **4 成的性能（0.68 → 0.38）是压缩瓶颈吃掉的**，
   而不是训练不充分、不是 prompt 格式、也不是 LoRA 能力问题。
4. **发现两个真实的数据管线 bug**，其中一个严重削弱了训练信号：
   - **D1（严重）**：question 右 padding 被夹在 question 和 answer 之间，导致**大多数训练样本
     的第一个 answer token 是在一个 padding 位置的 hidden state 上被监督的**。
   - **D2**：`append_eos` 的 `elif` 分支会把 batch 内最长答案的最后一个真实 token **覆盖成 EOS**；
     当 batch 内答案等长时（例如都是 1 个 token），**所有答案都会被覆盖成纯 EOS**。
5. **teacher-forced F1 不是一个可信的指标**：它衡量的是"给定答案前几个 gold token 后能否续写词形"，
   不是"能否从 memory 里取回答案"。D1 修复前 TF 首 token 准确率只有 3.0%，而自回归首词准确率
   有 26.0% —— TF 比 AR 还差，这本身就是 bug 的指纹。
6. **已用 4 卡对照实验定量确认了改进优先级**：修 D1/D2 → TF F1 **+0.128**；
   memory 8→64 → AR F1 **+0.110**；去掉重建损失 → AR F1 +0.032（不显著）。详见 §7。

---

## 1. 把 0.25 / 0.745 放到同一把尺子上

### 1.1 ICL baseline 的口径

`outputs/icl_baseline/qwen3-1.7b-0shot/squad.metrics.json`：`count=10570, em=0.6024, f1=0.7451`。
打分用的是 `src/icl_baseline.py:example_metrics`（官方 SQuAD 归一化 + **对全部 3~5 个参考答案取 max**），
prompt 是 chat template + 指令（"Answer each question using only the given passage. Return only a short
answer, without explanation."），输入上限 8192 token，left padding / left truncation。

### 1.2 用项目自己的度量重算 ICL baseline

用 `src/metrics.py:qa_metrics(pred, references[0])`（`normalize_text` 不去冠词、只用第一个参考）重算同一批预测：

| 打分口径 | F1 |
| --- | --- |
| ICL baseline 官方口径（max over refs） | **0.7451** |
| 项目自己的 `qa_metrics`，只取 `refs[0]` | **0.6810**（EM 0.4812） |
| 项目自己的 `qa_metrics`，max over refs | 0.7461 |
| 官方归一化，只取 `refs[0]` | 0.6837 |

53.2% 的 dev 样本有多个不同参考答案，max-over-refs 平均比 `refs[0]` 高 **+0.0613**。
而 `src/data.py:54 _answer()` 只保留第一个非空答案，评测端 `qa_metrics` 也只拿单个 reference。
**所以"0.25 vs 0.745"里有 ~6.4 分是纯口径差。**

### 1.3 评测集本身也不一致

- 报告里的 `0.2508` 来自 `validation_datasets: [squad]` + `validation_max_samples: 2000`，
  实际读到的是 `aggregated/squad/validation.jsonl`：**16498 条 QA = v1.1 的 10570 条 + v2.0 answerable 的 5928 条**。
- ICL baseline 只评 v1.1 的 10570 条。
- 也就是说模型的验证集混入了 SQuAD v2.0 那种对抗式构造的问题（更难），而 baseline 没有。
  两者不是同一个集合。

**结论：公平的当前对比是 0.3784（AR，本项目 checkpoint）vs 0.6810（ICL，同口径），约 0.56，而不是 1/3。**

---

## 2. 上界实验：瓶颈到底在哪

同一 checkpoint、同一批 300 个样本、同一 greedy 解码，只改"continuation 能看见什么"：

| 条件 | 说明 | project F1 | official F1 | official EM | 首词准确率 |
| --- | --- | --- | --- | --- | --- |
| `mem_tf` | 当前方法，teacher-forced argmax | 0.3082 | 0.3105 | 0.0300 | 0.040 |
| `full_tf` | 旁路 memory，TF argmax（完整 [ctx,q,a] 注意力） | 0.1762 | 0.1738 | 0.0033 | 0.010 |
| `mem_ar` | **当前方法，自回归** | 0.3479 | **0.3784** | 0.2567 | 0.260 |
| `full_ar` | **旁路 memory，自回归（上界）** | 0.6367 | **0.6847** | 0.5333 | 0.563 |
| `base_ar` | backbone（LoRA=0），同样的 raw 格式 | 0.1424 | 0.1613 | — | 0.027 |
| `base_icl_ar` | backbone（LoRA=0），ICL chat prompt | 0.6858 | 0.7407 | — | 0.500 |

可读出的结论：

1. **`full_ar = 0.6847` ≈ `base_icl_ar = 0.7407` ≈ ICL 的 0.7451。**
   同一个 LoRA 模型只要能看到完整 context，就能追平 ICL。**瓶颈是 memory 压缩，不是模型能力。**
2. **`mem_ar = 0.3784`，即 memory 前缀吃掉了 0.6847 → 0.3784 ≈ 0.31 F1（45% 相对损失）。**
3. `base_ar = 0.1613` 说明 raw prompt 格式对 **backbone** 很致命（0.16 vs 0.74），
   但 LoRA 训练已经把这个格式教会了（`full_ar` 0.68）。所以"格式"不是当前模型的问题，
   而是**如果将来换 backbone / 重新训练，格式必须先对齐**。
4. teacher-forced 与 autoregressive 完全不一致（`mem_tf` 0.31 < `mem_ar` 0.38），
   说明 TF 路径本身有问题（见 §3.1）。

---

## 3. 三个真实缺陷

### 3.1 D1（严重）：question 的右 padding 被夹在 question 和 answer 之间

`src/data.py:180` 用 `_encode(..., padding=True)` 对 question 做 **右 padding**；
`src/model.py:218 / 240` 再把 `[question, answer]` 直接 concat。
于是对一个 batch：

```
[context , memory , q0 q1 ... q_{v-1}  PAD ... PAD , a0 a1 ...]
                                        ^^^^^^^^^^^
        第一个 answer token a0 的 logits 来自最后一个 PAD 位置的 hidden state
```

`build_continuation_mask` 对 padding query 行只给了一条自环边（`src/model.py:139,147`），
所以那个位置的 hidden state 不携带任何 question/memory 信息。
**任何 question 长度 < batch 最大长度的样本，其第一个答案 token 的监督信号都是垃圾。**
训练时 `qa_per_context=4`、验证时 `sample_qa=False`（一个 context 的全部 QA 一起 padding），
所以**绝大多数样本都命中这个 bug**。

而自回归生成 `src/model.py:296` 用 `question_ids[row:row+1, :valid]` 截掉了 padding，
所以推理时第一个 token 来自真实的最后一个 question token —— **训练监督的位置和推理使用的位置不是同一个**。

复现（`.tmp_analysis/verify_bugs.py`，200 个样本）：

```
batch of 8 (questions right-padded)  project F1=0.2906  official F1=0.3056  首 token 准确率=0.030   (170/200 行 question 被 padding)
batch of 1 (no question padding)     project F1=0.4626  official F1=0.4742  首 token 准确率=0.320   (0/200 行被 padding)
```

受控实验（同一行样本，只改 batch 组成）：

```
short-question row: q_tokens=6   long-question row: q_tokens=26
  batched ALONE      -> first answer token pred='in'   gold='inf'  hit=False
  batched WITH long  -> first answer token pred='No'   gold='inf'  hit=False
  autoregressive     -> generated='infinite'          （正确）
```

**同一行样本，只要旁边放一个长问题，首 token 预测就从 `in` 变成 `No`。**
这一条几乎单独解释了"TF 首 token 准确率 3%"、"答案词形对但实体错"、
以及"自回归早期就崩"的全部现象。

**修复**：question 改为 left padding（`_encode` 时设 `padding_side="left"`），
`build_*_mask` 本来就是按 `q_valid` 位置索引的，天然兼容；
同时把 `src/model.py:296` 的 `question_ids[row, :valid]` 改成按 mask 取（`question_ids[row][question_mask[row]]`）。

### 3.2 D2：`append_eos` 覆盖最后一个答案 token

`src/data.py:181-186`：

```python
if valid < a.input_ids.size(1): a.input_ids[i, valid] = eos; ...
elif valid: a.input_ids[i, valid - 1] = eos      # <-- 覆盖最后一个真实答案 token
```

`padding=True` 会把 answer 右 padding 到 **batch 内最长**，所以 `valid == width` 的行就是
batch 内最长的答案 —— 这些样本的最后一个答案 token 被 EOS 直接抹掉。
如果 batch 内所有答案等长（短答案场景很常见），**所有答案都会被整条覆盖成 EOS**：

```
--- pad=right eos=overwrite
   ans tokens: ['<|im_end|>']            <-- 'cat' / 'dog' 整条丢失
   labels    : [151645]
--- pad=left eos=append
   ans tokens: ['cat', '<|im_end|>']     <-- 正确
   labels    : [4616, 151645]
```

**修复**：给 answer 张量多留一列，把 EOS 追加在 `valid` 位置（`eos_mode="append"`）。

### 3.3 D3：`0.0044` 是过期代码路径的历史数字；TF 指标本身误导

- `co5t73u7`（`0.0044` 那次）跑在 `7e2099e`，**早于 prefix-KV 重构**：旧的
  `MetaLoRA.forward` 是 `[context, memory, question, answer]` 单次联合前向，
  旧的 `generate_answer` 每一步都重算整个 context 编码且没有 KV cache。
  **当前代码 + 当前 checkpoint 重测 = official F1 0.3784。**
- 我另外验证了 `evaluation.max_new_tokens`（32/64/128）对结果**完全没有影响**
  （平均生成长度 5.4/5.5/5.8，F1 都是 0.3483）—— 所以 0.0044 也不是"生成太长被截断"造成的。
- `teacher_forced` 的语义问题：答案 token 是作为 **输入** 喂进去的，模型能从 gold 前缀续写词形。
  观察到的 TF 输出（`.tmp_analysis/diag_ar.py`）：
  `apoplectic stroke → "Nooplelectic"`、`Deabolis → "Nobal"`、`β-defensins → "V-defensin"`、
  `Catawba, Muskogee-speaking Creek and Choctaw → "Noreekwba, Muskogee,,, Choctaw"`
  —— **词尾常常完全正确，答案头部的实体完全错误**。这是"续写词形"而不是"检索"。
- 因此建议：**把 AR F1 作为主指标**，TF 只当作语言建模诊断量；
  另外单独上报"第一个答案 token 准确率 / answer span 命中率"作为检索能力的直接度量。

---

## 4. 结构性瓶颈：8 个 question-agnostic memory token

代码事实：

- `src/model.py:255-258`：prefix 编码后**只保留 memory 位置的 KV**（`keys[..., context_len:, :]`），
  continuation 的 attention 掩码（`build_continuation_mask`）**不允许 question/answer 看到任何 context token**。
- memory 在 **`encode_context_prefix`（只看 context）时就固定下来**，question 还没出现。
  `configs/qwen-1.7b/train.yaml:9` `memory_length: 8`。
- 训练集长度分布（`outputs/Qwen1.7B/sortish_lengths.json`，255143 个 context）：
  均值 177 token，p50 108，p90 433，max 2048。

即：把中位数 108、p90 433 token 的 context，压成 **8 个与问题无关的向量**，
再由 question 通过每层 8 路的 softmax 去取回答案。
上界实验显示这一层就损失了 0.31 F1。

---

## 5. 训练与工程问题清单

| # | 位置 | 问题 | 影响 |
| --- | --- | --- | --- |
| E1 | ~~`src/model.py:176`~~ **已修** | `self.to(dtype=qwen_dtype)` 原来把所有可训练参数都变成 bf16；现在新增 `model.trainable_dtype`（默认 `float32`）+ `MetaLoRA.set_trainable_dtype()`，并在 memory token 用作输入、LoRA 分支、decoder 入口/出口三处显式转换 | bf16 相对精度 ~2^-8，lr=1e-4 的 AdamW 更新会被量化：小模型实测一步后只有 **43/59** 个可训练张量发生变化，fp32 下是 **59/59** |
| E2 | `utils/config.py:174` | `validate()` 硬性要求 `max_context_tokens <= 2048` | 永远用不到 ICL baseline 的 8192 窗口 |
| E3 | ~~`scripts/train.py:81`~~ **已修** | 原来 `Accelerator(gradient_accumulation_steps=1)` 与 `cfg.training.grad_accumulation` 手动累加逻辑并存；现在 `grad_accumulation` 已从 config 与训练循环中移除，只保留 `Accelerator(gradient_accumulation_steps=1)` | 不再存在两套累加语义 |
| E4 | `src/model.py:68-149` | `build_block_causal_mask` / `build_continuation_mask` 是 O(B·T²) 的 Python 双重循环，并且物化 `[B,1,T,T]` 的 dense additive mask | 每步都重建；T=2048 时单个 mask 就是 B×8MB；是 OOM 的主要来源之一 |
| E5 | `src/model.py:243` | prefix 编码开 `output_hidden_states=True`，29 层 × 全 context 长度 的 hidden states 全部保留给 28 个 decoder | B×2048 时约 244MB/context；4 次 run 死于 OOM（峰值 21.7GB/23.5GB） |
| E6 | `utils/checkpoint.py:18` | `load_state_dict(..., strict=False)` | checkpoint 与模型不匹配时静默通过 |
| E7 | `src/evaluator.py:166` | `autoregressive()` 结尾又整跑一遍 `teacher_forced()` | 每次 AR 评测成本翻倍 |
| E8 | `src/losses.py:44-50` | `memory_contrastive_loss` 的 positive 项 `(z*z).sum(-1)/T ≡ 1/T` 是常数，实际只剩一个阈值式排斥项 | 语义与注释不符（weight=0 未启用）；该未启用目标随后已删除，见 `docs/READER_OPTIONS.md` |
| E9 | 环境 | `peft` 未安装，实际走 `StaticLoRALinear` 回退路径 | 与 PEFT 生态不兼容，且要求复现时保持同一路径 |
| E10 | `src/data.py:180` | answer 单独 tokenize 且 `add_special_tokens=False`，首 token 没有前导空格（`'Deabolis'→['De','abol','is']` vs `' Deabolis'→[' De','abol','is']`） | 训练目标形式不自然；影响有限但应统一 |

**数据配比**：`train_datasets: all` 过滤后 255143 个 context 中 **74.3% 是 ms_marco**，
squad 只占 7.4%（`outputs/Qwen1.7B/dataset_cache/train-*/hf_dataset`）。
而评测是 SQuAD。72% 的 context 只有 1 个 QA，mean 3.91，`qa_per_context=4` 实际几乎不产生多样性。

**训练量**：`co5t73u7` 只走了 37328/998811 步；`pgw1382s` 走到 20088/42524（3 卡 DDP 下相当于约 47% 个 epoch）。
所有 run 都远未充分训练。

---

## 6. 改进方案（按性价比排序）

### P0 —— 先修度量与 bug（几乎零成本，立即见效）

1. **修 D1**：question 改 left padding；`generate_answer_with_prefix` 按 mask 取 question token。
   *已落地并验证*：TF official F1 **+0.128**、TF EM **0.027 → 0.120**、首 token 准确率 **0.10 → 0.30**
   （4 卡对照实验，配对 bootstrap CI 不含 0）。**已应用到你仓库的代码里。**
2. **修 D2**：EOS 追加而不是覆盖（`eos_mode="append"`）。
3. **修度量**：AR F1 作为主指标；评测改 max-over-references + 官方归一化；
   另外上报 first-answer-token accuracy。
4. **修 D3**：把 `co5t73u7` 的 0.0044 作废，用当前代码重测；把 `evaluation.autoregressive_every`
   从 `1000000` 改回正常值（`pgw1382s` 整轮都没测过 AR）。

### P1 —— 打掉信息瓶颈（预期收益最大）

5. **memory_length 扫描**：8 → 32 → 64 → 128 → 256。成本几乎线性、改动只有一个数字。
   *已验证*：M 8→64 单独带来 AR official F1 **+0.110**（0.2534 → 0.3636，CI [+0.062,+0.158]）、
   AR EM **+0.103**、TF F1 **+0.057**。**这是所有改动里收益最大的一项**，继续往上扫还有空间。
6. **让 memory 条件化于 question**：两阶段编码（先 context→memory，再用 question 做一次
   cross-attention 重写 memory），或把 question 拼进 prefix 一起编码。
   当前 memory 在没见过 question 时就冻结了，这是检索失败的直接原因。
7. **混合方案 / 诚实上界**：保留 memory，同时允许 continuation 看一个滑窗或下采样后的 context KV；
   把 `full_ar = 0.6847` 当作 M→∞ 的端点，画"M vs F1"的压缩-精度曲线。
8. **memory 位置**：把 memory token 均匀插在 context 中间（gist/ICAE 风格），而不是全部堆在末尾。

### P2 —— 目标函数

9. ~~先做 `reconstruction_weight=0` 的对照~~ —— **结论已修正**：在 bf16 + 坏 target 格式下
   关掉重建项方向为正但不显著；**在 float32 + 修好的 target 格式下，关掉它反而使 AR F1 下降
   0.050（CI 不含 0）**。所以默认应**保留** `mse_cosine` 重建项（见 §9）。
10. **换重建目标（已实现）**：当前 target 是 **input embedding 序列**，等价于让 8 个向量去做"反 embedding"，
    几乎不可能学。已实现 `memory.reconstruction_loss: context_lm`：把每个 layer decoder 的
    bottleneck hidden 通过一个共享词表头（`Linear(D, vocab_size)`，39M，所有层共用）分类成 context token id，
    查询位置取 `t-1` 去预测 `t`，decoder 仍然只看 memory。两者都是 nats/token，权重可直接比较。
    实测代价 +16% 单步时间、+0.23 GiB 显存（见 README「Auxiliary memory objectives」）。
11. **加检索/复制辅助损失**：让 memory 与 question 的 cross-attention 直接指向答案 span
    （answer-span 位置监督），这是把"检索"变成显式监督最直接的办法。
12. ~~修 `memory_contrastive_loss` 的常数 positive 项，或换成 in-batch 对比（同 context 的
    question 为正、其他 context 为负）。~~ **该目标已删除**（未启用的死选项 + 语义有缺陷）；
    若日后仍想做"同 context 为正、跨 context 为负"的对比，请按 `docs/READER_OPTIONS.md`
    的口径重新加一个带测试的新目标，而不是复活这个实现。

### P3 —— 训练与工程

13. **可训练参数保持 fp32**（backbone 仍 bf16）—— **已实现**，见 `model.trainable_dtype`。
    三个转换点：`memory_tokens.to(dtype=self.dtype)`（可微）、`StaticLoRALinear` 在 adapter
    dtype 下算 delta 再转回、`MemoryDecoder` 入口转 fp32 出口返回 fp32，loss 端统一 upcast 到
    fp32 归约。autocast 由 `src/dtypes.py:no_autocast` 在 adapter/decoder 内部关闭。参数规模
    约 8×10⁷，显存代价约 +0.8GB，需配合关掉重建 decoder / gradient checkpointing。
    旧的 bf16 checkpoint 仍可直接加载（`load_state_dict` 会转换）。
14. 向量化两个 mask。**已落地到 `src/model.py`**（`torch.equal` 与旧实现逐元素相等，
    见 `.tmp_analysis/verify_masks.py`）：L=2048/B=2 时 492.55 ms → 0.76 ms（648×），
    短样本 L=128 也由 61.40 ms → 0.62 ms。根因是旧实现每行读一次 GPU 标量
    （`if c_valid[i]:`）导致 ~0.24 ms/token 的设备同步。详见 `ENGINEERING.md` §1。
    进一步可拆成「context causal pass + 8-query memory pass」，让大前向彻底不需要自定义 mask。
15. 显存：`gradient_checkpointing` 已在 config 里但从未接线；decoder 只需要 memory 位置的
    hidden state，可以避免保留 29×L 的完整 hidden states。
16. 修 `strict=False`、grad accumulation 不一致、`autoregressive` 重复调用 TF。
17. 提高 `max_context_tokens` 上限（配合 gradient checkpointing）。

### P4 —— 数据与协议

18. 训练配比重平衡：74% ms_marco vs SQuAD 评测；至少分数据集上报指标。
19. 早期对比建议先用 `train_datasets: [squad]`，把"压缩瓶颈"从"领域不匹配"里分离出来。
20. **公平对比**：ICL baseline 也用项目自己的度量（0.6810）；模型也在 v1.1 子集上报一次；
    再补一个"同一 raw prompt 格式的 backbone"基线（本次测得 0.1613）。

---

## 7. 4 卡对照实验（已完成）

4 张 4090 并行训练 4 个 arm，每个 6000 步、batch 1、**同一份 SQuAD-train 子集、同一 seed、
同一超参**，唯一区别如下；训练完在同一批 300 个 SQuAD v1.1 dev 样本上评 TF + AR（greedy，max_new 32）。

| arm | question padding | EOS | M | recon_weight |
| --- | --- | --- | --- | --- |
| `armA_control` | right（现状） | overwrite（现状） | 8 | 1.0 |
| `armB_fixed` | left | append | 8 | 1.0 |
| `armC_fixed_mem64` | left | append | 64 | 1.0 |
| `armD_fixed_norecon` | left | append | 8 | 0.0 |

### 7.1 结果

| arm | TF 首 token 准确率 | TF official F1 | TF official EM | AR official F1 | AR official EM | AR 首词准确率 |
| --- | --- | --- | --- | --- | --- | --- |
| `armA_control` | 0.100 | 0.2466 | 0.0267 | 0.2284 | 0.1200 | 0.177 |
| `armB_fixed` | **0.300** | **0.3750** | **0.1200** | 0.2534 | 0.1333 | 0.190 |
| `armC_fixed_mem64` | **0.363** | **0.4321** | **0.1767** | **0.3636** | **0.2367** | **0.273** |
| `armD_fixed_norecon` | 0.300 | 0.3824 | 0.1267 | 0.2850 | 0.1633 | 0.213 |

### 7.2 配对 bootstrap 检验（n=300，4000 次重采样）

| 对比 | 指标 | Δ | 95% CI | better/worse/same |
| --- | --- | --- | --- | --- |
| `armB_fixed − armA_control`（修 D1+D2） | TF F1 | **+0.1284** | [+0.093, +0.164] | 94/24/182 |
| `armB_fixed − armA_control` | AR F1 | +0.0250 | [−0.010, +0.061] | 51/46/203 |
| `armC_fixed_mem64 − armB_fixed`（M 8→64） | TF F1 | **+0.0572** | [+0.020, +0.094] | 80/42/178 |
| `armC_fixed_mem64 − armB_fixed` | **AR F1** | **+0.1102** | **[+0.062, +0.158]** | **89/38/173** |
| `armD_fixed_norecon − armB_fixed`（去掉重建） | TF F1 | +0.0075 | [−0.019, +0.036] | 37/43/220 |
| `armD_fixed_norecon − armB_fixed` | AR F1 | +0.0316 | [−0.002, +0.066] | 49/35/216 |

### 7.3 结论

1. **M 8→64 是单变量收益最大、统计上最扎实的一项**：
   AR official F1 **+0.110**（0.2534 → 0.3636，CI 不含 0），AR EM **+0.103**（0.133 → 0.237），
   首 token / 首词准确率同时提升（0.30 → 0.363 / 0.190 → 0.273）。
   这与 §2 的上界实验（full_ar 0.6847 vs mem_ar 0.3784）完全一致：**瓶颈就是 memory 容量**。
   继续往上扫 128 / 256 是性价比最高的下一步。
2. **修 D1+D2 让 TF 指标从"不可信"变成"可信"**：TF official F1 **+0.128**、EM **+0.027 → +0.120**、
   首 token 准确率 **0.10 → 0.30**，全部远超噪声。AR 上方向一致（+0.025，EM +0.013）但 n=300 下不显著；
   结合 §3.1 的受控复现，这两个 bug 仍是必须修的（它们让训练监督指向错误位置）。
3. **去掉重建损失方向为正但不显著**（AR +0.032，CI 下界 −0.002），而重建 loss 本身早已不下降。
   建议默认关掉（同时省显存/时间），但不要指望它是主要收益来源。
4. 四个 arm 都在 6000 步、只有 6000 个 SQuAD context 的条件下训练，因此绝对值低于
   §2 里 20k 步、全量数据混合的 checkpoint；**这里只看相对差值**。
5. 即便如此，`armC` 的 AR 0.3636 仍显著低于 `full_ar` 的 0.6847 —— **压缩瓶颈远未被解决**。

产物：`.tmp_analysis/exp/run_all.sh`（脚本）、`.tmp_analysis/exp/logs/*.log`（日志）、
`.tmp_analysis/exp/results/*.json`（逐样本预测，可直接复算）。

---

## 8. 已落地的修复

`src/data.py`、`src/model.py`、`utils/config.py`、`scripts/train.py`、`scripts/test.py` 已应用以下改动
（`git diff` 可查，可用配置回退到旧行为）：

1. `data.question_padding_side`（默认 **`left`**）：question 改为 left padding，
   使真实 question token 紧邻 answer；设为 `right` 可复现旧 run。
2. `data.eos_mode`（默认 **`append`**）：EOS 追加在最后一个答案 token 之后；设为 `overwrite` 复现旧行为。
3. `generate_answer_with_prefix` 改为按 `question_mask` 取 question token，兼容两种 padding。
4. 旧行为下的两个 bug 都有最小复现（`overwrite` 会把整条答案变成 `<|im_end|>`；
   同一行样本换个 batch 邻居就改变首 token 预测）。
5. 已通过 CPU 端到端 smoke test：collate / forward / backward（所有可训练参数都有有限梯度）/ generate。


---

## 9. 第二批 4 卡实验：可训练参数 dtype（float32 vs bfloat16）

固定 `left` padding + `append` EOS + M=8 + recon 1.0，**只改 `model.trainable_dtype`**，
各 6000 步、同一份 SQuAD-train 子集、同一 seed。评测同一批 300 个 SQuAD v1.1 dev 样本。

| arm | trainable_dtype | recon_weight | TF official F1 | TF official EM | TF first-token | AR official F1 | AR official EM | AR 首词 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `armE_fp32_M8` | **float32** | 1.0 | **0.4364** | **0.1700** | **0.323** | **0.3497** | **0.2267** | 0.250 |
| `armF_bf16_M8` | bfloat16 | 1.0 | 0.3683 | 0.1200 | 0.297 | 0.2863 | 0.1700 | 0.220 |
| `armH_fp32_norecon` | float32 | **0.0** | 0.4047 | 0.1567 | 0.293 | 0.2996 | 0.1867 | 0.223 |

配对 bootstrap（n=300，4000 次重采样）：

| 对比 | 指标 | Δ | 95% CI |
| --- | --- | --- | --- |
| fp32 − bf16 | AR F1 | **+0.0635** | [+0.021, +0.105] |
| fp32 − bf16 | AR EM | **+0.0567** | [+0.017, +0.097] |
| fp32 − bf16 | TF F1 | **+0.0681** | [+0.038, +0.098] |
| fp32 − bf16 | TF EM | **+0.0500** | [+0.020, +0.080] |
| 去掉 recon − 保留 recon（都 fp32） | AR F1 | **−0.0501** | [−0.092, −0.007] |
| 去掉 recon − 保留 recon（都 fp32） | AR EM | −0.0400 | [−0.083, +0.000] |

结论：

1. **可训练参数保持 float32 带来统计显著的一致提升：AR F1 +0.064、AR EM +0.057、
   TF F1 +0.068、TF EM +0.050，四个指标的 CI 都不含 0。** 机制见 §8 与 README
   "Two dtypes"：bf16 下 lr=1e-4 的 AdamW 更新会被量化掉（小模型实测一步后仅 43/59 个张量变化，
   fp32 下 59/59）。
2. **重建损失不该关掉**：在 fp32 + 修好 target 格式之后，去掉重建损失反而使 AR F1 下降 0.050
   （CI 不含 0）。这与第一批实验（bf16、buggy 格式下 +0.032 且不显著）方向相反 ——
   说明之前"reconstruction loss 是死权重"的结论**只对 bf16 + 坏 target 的那个配置成立**。
   §6 的 P2 第 9 条据此修正：**保留 MSE+cosine 重建项**；若要换成 `context_lm`（见下）需要用同样的
   对照实验验证。
3. 说明参数精度和判别目标是耦合的：参数能真正更新之后，重建目标才开始提供有用的辅助信号。

产物：`.tmp_analysis/exp2/run_all.sh`、`eval_only.sh`、`logs/*.log`、`results/*.json`。

## 10. 修复清单补充

- `model.trainable_dtype`（默认 `float32`）+ `MetaLoRA.set_trainable_dtype()` + `src/dtypes.py:no_autocast`。
- `qa_loss` / `reconstruction_loss` 统一在 float32 下归约。
- `is_trainable_parameter_name()` 作为"哪些参数可训练"的唯一来源，`set_trainable_dtype` 与
  `load_model` 共用，避免新增模块时漏改一处。
- 旧的 bf16 checkpoint 可直接加载进 fp32 参数（`load_state_dict` 会转换），已验证 813 个可训练张量全部保持 float32。

---

## 11. 4000 步崩溃：NCCL watchdog 超时（rank-0-only 评测）

### 现象

```
Training: 6%|...| 4000/63786 [42:35<10:04:20, 1.65step/s, loss=10.3148, qa=2.1608, recon=8.1540
[rank2][E] Watchdog caught collective operation timeout:
    WorkNCCL(SeqNum=40012, OpType=BROADCAST, NumelIn=128, Timeout(ms)=600000)
```

只有 rank 1/2/3 报超时，rank 0 没有。

### 先排除一个误报：`recon=8.1540` 不是发散

`reconstruction_loss` 现在是 `context_lm`，它是**每 token 的交叉熵（nats）**。
随机初始化时 `ln(151936) ≈ 11.93`，8.15 是正在正常下降的 CE。
`loss = qa + recon = 2.16 + 8.15 = 10.31` 完全对得上。不要按数值大小判断发散。

### 根因

`accelerator.prepare` 会把验证集按 rank 切片（`BatchSamplerShard`），而 `scripts/train.py` 里评测写的是：

```python
if is_main and step % cfg.evaluation.autoregressive_every == 0:
    metrics = evaluator.autoregressive(...)
```

**只有 rank 0 做自回归评测。** rank 0 每步自回归解码
`validation_max_samples 2000 / world_size 4 ≈ 500` 个 context × 约 8 个 QA ≈ **4000 行**，
每行最多 32 个 token（实测约 0.28 s/行）→ **约 19 分钟**，
远超 NCCL 看门狗的 10 分钟（600000 ms）。其余 rank 直接进入下一个训练步，
卡在 DDP 开头的 128 元素 BROADCAST 上干等 → 超时后所有 rank 被 `SIGABRT` 带走。

这条路径之前一直没暴露，是因为：`nhujmefn` / `p5x4m7sd` 的自回归评测在 39% 时被 Ctrl-C，
`pgw1382s` 干脆把 `autoregressive_every` 设成 `1000000` 关掉了
——**多卡 DDP 下自回归评测从来没有真正跑完过。**

### 顺带暴露的第二个缺陷：验证指标只覆盖 rank 0 的分片

既然 `prepare` 已经切分了验证集，而评测只让 rank 0 跑，
那么所有多卡 run 记录的 `val/teacher_forced/f1` 其实是
"每 `world_size` 个 context 取一个"的约 1/4 子集，**不是整个验证集**。
不同 `world_size` 的 run 之间因此不可比（1 卡 run 是整个集合，4 卡 run 是 1/4 步长子集）。

### 修复（已落地）

| 改动 | 位置 |
| --- | --- |
| 评测改为**每个 rank 都跑自己的分片**，累积量用 `_distributed_sum` 做 all_reduce(SUM) 后再相除；`is_main` 只用于 `run.log` | `src/evaluator.py`、`scripts/train.py` |
| 新增 `evaluation.autoregressive_max_qa`（默认 256，**每 rank**），把自回归评测的绝对耗时压进看门狗窗口；全局解码行数 = 该值 × `world_size` | `utils/config.py`、三个 `train.yaml` |
| 评测块之后加 `accelerator.wait_for_everyone()` 显式同步 | `scripts/train.py` |

### 验证

1. **2 进程 DDP 冒烟**：4 步训练，step 2/4 触发 TF、step 4 触发 AR，全部跑完，无超时。
2. **all-reduce 正确性**：同一 seed、同一验证集（8 个 context），
   `--num_processes 1` 与 `--num_processes 2` 的指标完全一致：

   | world_size | TF F1 | AR F1 |
   | --- | --- | --- |
   | 1 | 0.148543 | 0.010509 |
   | 2 | **0.148543** | **0.010509** |

   脚本：`.tmp_analysis/ddp_eval_check.py`。

### 还有哪些 rank-0-only 操作会踩同一个坑

- `manager.save(...)` 只在 rank 0 执行，其余 rank 在下一个 allreduce 等它。
  当前 checkpoint 约 700MB（fp32 可训练参数），写盘几十秒，安全；
  但如果 checkpoint 继续变大或磁盘变慢，同样会撞 10 分钟看门狗。
- 首次构建 255k context 的 dataset cache 时是一个 rank 建、其余用 flock 等，
  这发生在 `prepare` 的第一次集合通信之前；若耗时超过 10 分钟也会报同样的错。
  建议先用单进程预热一次缓存。
