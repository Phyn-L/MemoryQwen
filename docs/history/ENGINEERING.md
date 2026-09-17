# 工程问题：根因、证据与修复

> 历史快照：2026-09-18 归档。正文状态、数值、路径和预计完成时间属于当时记录，本次未重新运行实验或核实远端状态。当前操作见[运行指南](../guides/RUNNING.md)，实验状态见[实验索引](../experiments/README.md)。

所有数字都是在本机 4×4090（24 GiB）上实测的，脚本在 `.tmp_analysis/bench_eng.py`、
`.tmp_analysis/bench_attn.py`、`.tmp_analysis/bench_attn_one.py`、`.tmp_analysis/bench_lmhead.py`、
`.tmp_analysis/bench_mask_sweep.py`。

## 现状一览

| 问题 | 状态 |
| --- | --- |
| §1 O(B·T²) Python mask | **已修**（`src/model.py` 已换成向量化实现，`torch.equal` 与旧实现逐元素相等） |
| §2 gradient_checkpointing 未接线 | **不改**（配置键已删除，不再有"设了不生效"的陷阱）：prefix KV 复用是 context-level 聚合的设计前提，与 HF checkpointing 天然冲突（见下） |
| §3 checkpoint `strict=False` | **已修**：保留 `strict=False`（冻结 backbone 本来就不存），补上「可训练子集必须完整加载」的校验 |
| §4 AR 重复跑 TF | **已修**：不再重复跑 TF，且同一 context 的多个 QA 已合并成一个 batch 做 prefill |
| §5 Accelerator 与 grad_accumulation 不一致 | **已修**：gradient accumulation 机制整体切除，每个 DataLoader batch 一次 optimizer step |
| §6 `max_context_tokens <= 2048` | **不改**（有意的成本控制），只补了过滤计数打印 |
| §7 prefix pass 白算 LM head | **已修**：prefix pass 改走 `_transformer_body`，输出逐位一致、前向快 3.6×、省 596 MB/行；`_encode_context` 死代码已删 |

---

## 1. 两个 O(B·T²) 的 Python mask

### 现象 / 实测

`B=2, M=8, QL=64, AL=32`，`build_block_causal_mask` 每次调用的耗时：

| context 长度 L | 现状 | 向量化后 | 倍数 |
| --- | --- | --- | --- |
| 128 | 61.40 ms | 0.62 ms | 99× |
| 256 | 86.88 ms | 0.88 ms | 99× |
| 512 | 141.75 ms | 0.70 ms | 203× |
| 1024 | 261.82 ms | 0.74 ms | 354× |
| 2048 | 492.55 ms | 0.76 ms | 648× |

`build_continuation_mask`（B=8, M=8, QL=64, AL=32）：**104.95 ms → 0.46 ms（228×）**。

对照：4 卡对照实验实测训练吞吐是 **2.27 step/s ≈ 440 ms/step**。也就是说
**光是构造 block mask 在 L=2048 时就要 493 ms，比整个训练 step 还慢**；即使在 L=128
的短样本上也要 61 ms，占掉 step 时间的 ~14%。这两个函数是当前最大的单点性能损失。

### 根因（三层，逐层放大）

1. **Python 双重循环 + 每行一次设备同步。**
   `src/model.py:100-124`：

   ```python
   for batch in range(bsz):
       c_valid = context_mask[batch]          # GPU tensor
       ...
       for i in range(context_len):
           if c_valid[i]:                     # <-- 读 GPU 标量 -> 隐式 .item() -> cudaStreamSynchronize
               allowed[batch, c0 + i, c0 : c0 + i + 1] = c_valid[: i + 1]
           else:
               allowed[batch, c0 + i, c0 + i] = True
   ```

   `if c_valid[i]:` / `if q_valid[i]:` / `if a_valid[i]:` 都在对 GPU 张量取布尔值，
   每次触发一次 D2H 拷贝 + 流同步；再加上每个切片赋值都是一次独立的 kernel launch。
   L=2048、B=2 时大约有 `2 × (2048 + 64 + 32) ≈ 4300` 次同步/launch。
   实测斜率约 **0.24 ms / context token**，与「每 token 一次同步」完全吻合。

2. **物化 O(B·T²) 的稠密加性 mask。**
   `torch.zeros((bsz, 1, total, total), dtype=bfloat16)` + `masked_fill`。
   T = L + M + QL + AL，L=2048 时是 2×1×2312²×2 B ≈ 21 MB，每步都重新分配。
   张量本身不算致命，但它决定了下一步。

3. **4 维加性 mask 会把 SDPA 从 flash 路径踢到 math 路径。**
   这一点要分两种情况说清楚，实测结论是「不能只靠传 2 维 mask 解决」：

   | 变体（L=2048, B=1） | peak |
   | --- | --- |
   | 2 维 padding mask | 14.00 GiB |
   | `attention_mask=None` | 14.00 GiB |
   | 4 维加性 causal mask | 14.23 GiB |

   三者几乎一样。原因是 transformers 的 `_update_causal_mask` 在 sdpa 后端下
   **本来就会把 2 维 mask 展开成 4 维加性 causal mask**（除非 `attn_implementation="flash_attention_2"`）。
   所以「改传 2 维 mask」不会自动恢复 flash 内核。真正要解决它，必须让**大**的那次前向
   不再需要自定义 mask（见下面第 3 条修复方案）。

### 修复

**(a) 直接替换成向量化版本（已落地到 `src/model.py`，并验证与旧实现逐元素完全相等）**

在 30 组随机 mask（随机 batch 数、context 长度、question/answer 长度，含全 padding 行）
上 `torch.equal(old, new)` 全部通过；`.tmp_analysis/verify_masks.py` 里的参照实现
直接从 `git show HEAD:src/model.py` 里 exec 出来，不依赖我手抄。当前 `src/model.py` 已是向量化版本：

```python
def build_block_causal_mask(context_mask, memory_length, question_mask, answer_mask, dtype):
    """Vectorised equivalent of the original row-by-row builder.

    Context rows are causal over valid context tokens; memory rows see every valid
    context token and nothing else; question rows see memory plus a causal view of
    valid question tokens; answer rows additionally see every valid question token.
    Padding query rows keep a single self edge so no softmax row is fully masked.
    """
    c, q, a = context_mask.bool(), question_mask.bool(), answer_mask.bool()
    bsz, length = c.shape
    qlen, alen = q.shape[1], a.shape[1]
    total = length + memory_length + qlen + alen
    mem_start, q_start, a_start = length, length + memory_length, length + memory_length + qlen
    dev = c.device
    allowed = torch.zeros(bsz, total, total, dtype=torch.bool, device=dev)
    causal_ctx = torch.tril(torch.ones(length, length, dtype=torch.bool, device=dev))
    allowed[:, :length, :length] = c[:, None, :] & causal_ctx
    allowed[:, mem_start:q_start, :length] = c[:, None, :]
    causal_qa = torch.tril(torch.ones(qlen + alen, qlen + alen, dtype=torch.bool, device=dev))
    allowed[:, q_start:a_start, mem_start:q_start] = True
    allowed[:, q_start:a_start, q_start:a_start] = q[:, None, :] & causal_qa[:qlen, :qlen]
    allowed[:, a_start:, mem_start:q_start] = True
    allowed[:, a_start:, q_start:a_start] = q[:, None, :]
    allowed[:, a_start:, a_start:] = a[:, None, :] & causal_qa[qlen:, qlen:]
    ctx_rows = torch.arange(length, device=dev)
    ctx_eye = torch.zeros(bsz, length, total, dtype=torch.bool, device=dev)
    ctx_eye[:, ctx_rows, ctx_rows] = True
    allowed[:, :length] = torch.where(c[:, :, None], allowed[:, :length], ctx_eye)
    qa_rows = torch.arange(qlen + alen, device=dev)
    qa_eye = torch.zeros(bsz, qlen + alen, total, dtype=torch.bool, device=dev)
    qa_eye[:, qa_rows, q_start + qa_rows] = True
    qa_valid = torch.cat([q, a], dim=1)
    allowed[:, q_start:] = torch.where(qa_valid[:, :, None], allowed[:, q_start:], qa_eye)
    mask = torch.zeros(bsz, 1, total, total, dtype=dtype, device=dev)
    return mask.masked_fill(~allowed[:, None], torch.finfo(dtype).min)


def build_continuation_mask(question_mask, answer_mask, memory_length, dtype):
    """Vectorised equivalent for the [question, answer] continuation over a memory cache."""
    q, a = question_mask.bool(), answer_mask.bool()
    bsz, qlen = q.shape
    alen = a.shape[1]
    cur = qlen + alen
    total = memory_length + cur
    dev = q.device
    idx = torch.arange(cur, device=dev)
    causal = idx[:, None] >= idx[None, :]
    allowed = torch.zeros(bsz, cur, total, dtype=torch.bool, device=dev)
    allowed[:, :, :memory_length] = True
    allowed[:, :qlen, memory_length:memory_length + qlen] = q[:, None, :] & causal[:qlen, :qlen]
    allowed[:, qlen:, memory_length:memory_length + qlen] = q[:, None, :]
    allowed[:, qlen:, memory_length + qlen:] = a[:, None, :] & causal[qlen:, qlen:]
    valid = torch.cat([q, a], dim=1)
    self_edge = torch.zeros_like(allowed)
    self_edge[:, idx, memory_length + idx] = True
    allowed = torch.where(valid[:, :, None], allowed, self_edge)
    mask = torch.zeros(bsz, 1, cur, total, dtype=dtype, device=dev)
    return mask.masked_fill(~allowed[:, None], torch.finfo(dtype).min)
```

只改这两个函数、不改语义，就能把 mask 开销从「与 step 同量级」降到噪声级别。

**(b) 更好：让大的那次前向彻底不需要自定义 mask**

`encode_context_prefix` 现在把 `[context, memory]` 放在一次前向里，才需要 `build_block_causal_mask`。
但注意 mask 的语义其实是「context 走 causal；memory 是 8 个 query，双向看 context，彼此不可见」——
这可以拆成两次前向：

1. **context pass**：`attention_mask = context_mask`（2 维）+ causal，**不**要自定义 mask、
   **不**要 `output_hidden_states`，`use_cache=True` 拿到 KV cache；
2. **memory pass**：把 `memory_tokens` 当作 8 个 query，`past_key_values` 接上一步的 cache，
   `attention_mask` 是 `[B, 1, M, L+M]` 的 4 维 mask（只有 8 行，走 math 路径也无所谓），
   `output_hidden_states=True` 拿到各层 memory 隐状态给 decoder。

收益：
- 大的那次前向变成纯 causal，没有 `[B,1,T,T]` 物化、没有 Python 循环、SDPA 可以走非 math 内核；
- decoder 需要的 `layer_memory` 变成 `29 × B × M × H`（M=8）而不是从整段
  `[B, L+M, H]` 里切，语义等价但更干净。

**(c) 顺手**：`scripts/train.sh` 里加 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
历史上 `pgw1382s` / `3d9ya35y` 的 OOM 报错自己就提示了这一条，但脚本里没有。

### 附：全遮蔽行会不会 NaN？——不会，但原因是"写的是 finfo.min 而不是 -inf"

审计时一度怀疑 memory 行存在 NaN 隐患：memory 行只 attend context，若某行 context 全为
padding，这 M 行就没有任何 key。实测结论是**不会 NaN**，机制值得记下来：

- `build_block_causal_mask` 用 `masked_fill(~allowed, torch.finfo(dtype).min)`，写入的是**有限**的
  最负值，不是 `-inf`；全遮蔽行因此是"所有 logits 相等"，softmax 退化成均匀分布，有限。
- 直接对真实 Qwen3 attention 注入"整行遮蔽"的 mask 验证（eager 与 sdpa 都试）：
  用 `finfo.min` 两种后端都有限；把同一行的值换成 `-inf`，**eager 立刻 NaN**，sdpa 仍然有限。
  也就是说"不会 NaN"完全依赖于那个有限值，谁把它"简化"成 `-inf` 就会踩雷。
  `tests/test_masks.py::test_blocked_entries_are_finite_so_no_row_can_nan` 把这一点钉住了。
- 不过均匀分布仍然是无意义的（memory 隐状态会被 decoder 和 prefix 使用），所以真正的防护在
  数据层：`_build_records` 现在除了丢弃空字符串 context，也会丢弃**分词后为空**的 context，
  使"context 行至少有一个有效 token"成为代码保证而非注释假设。

---

## 2. gradient_checkpointing 没有接线（而且直接开是坏的）

### 现象

- `ModelConfig.gradient_checkpointing: bool = False` 曾经定义在 `utils/config.py`，
  **`load_model` 从来没有读过它**，代码里也没有任何 `gradient_checkpointing_enable()`。
  该配置键已在死配置清理中删除（`encoder_heads` / `freeze_backbone` / `test_max_samples`
  同样是没有任何读取方的残留），所以现在连"设了却不生效"的陷阱也不存在了。
- 但直接打开会直接报错。实测：

  ```
  [transformers] `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
  RuntimeError: Qwen did not return a KV cache for prefix encoding
  ```

### 根因

1. **配置项是死代码。** `load_model()` 构造 `MetaLoRA` 时只传了 `lora_*` / `memory_*` /
   `target_modules` / `max_context_tokens`，没有任何一行读 `cfg.model.gradient_checkpointing`。
2. **架构和 HF 的 gradient checkpointing 直接冲突。** HF 的 `Qwen3Model.forward` 里有

   ```python
   if self.gradient_checkpointing and self.training and use_cache:
       logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
       use_cache = False
   ```

   而 `encode_context_prefix`（`src/model.py:293`）用 `use_cache=True` 并依赖
   `out.past_key_values` 来构造 memory KV cache（`src/model.py:295` 起）。
   于是 checkpointing 一开，cache 变 `None`，直接抛
   `RuntimeError("Qwen did not return a KV cache for prefix encoding")`。
   **也就是说，prefix-sharing 这个设计和「对整个 backbone 开 gradient checkpointing」是不兼容的**，
   不是加一行 `gradient_checkpointing_enable()` 就能解决。
3. **frozen backbone + `inputs_embeds` 的经典坑。** backbone 全部 `requires_grad=False`，
   `inputs_embeds` 由 frozen embedding 产生（`requires_grad=False`），
   reentrant checkpointing 下会出现 "None of the inputs have requires_grad=True"，
   梯度断在 checkpoint 段里。需要 `enable_input_require_grads()` 或对 `inputs_embeds`
   手动 `requires_grad_(True)`。
   注意这里 embedding 是在 `MetaLoRA.forward` 外面算好再传进来的，
   所以 HF 的 `enable_input_require_grads()` 挂的 hook **根本不会被触发**。

### 结论：不改，保持 prefix KV 复用

数据按 context level 聚合，所以「context 只前向一次得到 KV cache，之后该 context 的所有 QA
直接复用」是这套设计的核心收益（同一 context 的 11 个 QA 只编码 context 一次）。
`use_cache=True` 因此是**必需**的，而 HF 的 gradient checkpointing 会强制把它关掉 ——
两者不可兼得。既然 prefix KV 复用不能放弃，**结论就是不对 backbone 开 checkpointing**，
显存改从别处省：

| 手段 | 实测收益（L=2048, B=1） |
| --- | --- |
| prefix pass 改走 `self.qwen.model`（不算 LM head） | 8.28 → 7.70 GiB（−0.58 GiB/行） |
| `reconstruction_weight == 0` 时不跑 28 个 decoder | 省掉 decoder 前向 + 约 0.5 GiB 输出/张量 |
| 按 **context token 预算** 组 batch，而不是固定 batch size | sortish sampler 会把长样本聚在一起，峰值由最长 batch 决定；这是 OOM 的直接来源 |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 历史 OOM 日志自己就提示过，`scripts/train.sh` 里没有 |

如果将来确实还需要 checkpointing，唯一可行的是**手写 prefix 段的逐层 checkpoint**：
自己遍历 `self.qwen.model.layers`，对每层用 `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`
并自己累积 `DynamicCache`。注意 HF 的 `Qwen3DecoderLayer` 在 `use_cache=True` 时返回
`(hidden_states, present_key_value)`，所以 cache 是拿得到的；坑在于 checkpoint 的第二个
返回值是 Cache 对象而不是张量。另外无论哪种方案，`inputs_embeds` 都在 `MetaLoRA.forward`
外面算好，所以 HF 的 `enable_input_require_grads()` hook 不会被触发，需要自己
`requires_grad_(True)`。

## 3. `checkpoint.py:18` 的 `strict=False`

### 现象

```python
missing = model.load_state_dict(state.get("model", {}), strict=False)
```

返回值被丢掉，加载永远「成功」。

### 根因

`CheckpointManager.save` 只保存 `requires_grad` 的参数：

```python
trainable = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
```

冻结的 1.7B backbone 不在 checkpoint 里，所以 `strict=True` 会因为几百个
"missing keys" 直接失败。作者为了绕开这个，改成了 `strict=False` —— 结果把**所有**
校验一起丢掉了。

后果（都是静默的）：

- 用 `memory.memory_length: 8` 的 checkpoint 去初始化 `memory_length: 64` 的配置：
  只有 8 行 `memory_tokens` 被加载，其余 56 行保持 `randn * 0.02` 的随机初始化。
  模型照常训练、照常出 loss，只是有 7/8 的 memory 从来没被训过。
- 改 `decoder_hidden_size` / `decoder_heads` / `lora_rank` / `target_modules` 同理：
  形状不匹配的张量全部被跳过。
- `lora_B` 是零初始化（`nn.init.zeros_`），所以「LoRA 完全没加载上」在第 0 步
  和「LoRA 正常」在行为上完全一样，很难靠观察发现。

### 修复（已落地到 `utils/checkpoint.py`）

`strict=False` 本身是对的 —— 冻结的 backbone 本来就不该进 checkpoint，
所以 `strict=True` 会因为几百个 missing keys 直接失败。问题只在于**返回值被丢掉了**。
现在保留 `strict=False`，但补上三道校验，判据都只落在「你实际保存的那部分」上：

```python
try:
    missing, unexpected = model.load_state_dict(saved, strict=False)
except RuntimeError as exc:                      # 形状不匹配（torch 自己就会报）
    raise RuntimeError(f"checkpoint {path} does not fit this model: {exc}\n"
                       "A shape-defining config value (memory_length, lora_rank, "
                       "decoder_hidden_size/decoder_heads/decoder_ffn_ratio, target_modules) "
                       "differs from the run that produced it.") from exc

if unexpected:                                   # 文件里有模型用不上的张量
    raise RuntimeError(f"checkpoint {path} holds {len(unexpected)} tensors this model has no "
                       f"place for, e.g. {sorted(unexpected)[:8]}.")

trainable = {n for n, p in model.named_parameters() if p.requires_grad}
absent = sorted(trainable.intersection(missing))  # 可训练子集必须完整
if absent:
    detail = (f"checkpoint {path} is missing {len(absent)} of {len(trainable)} trainable "
              f"tensors, e.g. {absent[:8]}. They would silently keep their random initialisation.")
    if not allow_missing_trainable:
        raise RuntimeError(detail + " Pass allow_missing_trainable=True to load the rest anyway.")
    print(f"[checkpoint] WARNING: {detail}")
```

关键点：`missing` 里剩下的只会是冻结的 backbone，所以 `trainable ∩ missing` 既精确又不会误报。
`allow_missing_trainable=True` 是给「用旧 checkpoint 做初始化」留的显式出口，
`scripts/test.py` 也加了对应的 `--allow-missing-trainable` 开关。

实测四种情形（`.tmp_analysis/verify_checkpoint.py`）：

| 情形 | 结果 |
| --- | --- |
| 同构模型加载 | 正常，`step=7` |
| `memory_length` 8 → 16 | `RuntimeError`：指出是 shape-defining 配置变了 |
| checkpoint 少一个可训练张量 | 默认 `RuntimeError`；加 flag 则 WARNING 后继续 |
| 文件里多一个陌生张量 | `RuntimeError`：列出具名张量 |

## 4. AR 评测：不再重复跑 TF，且同一 context 的多个 QA 合并成一个 batch

### 4.1 重复跑 TF —— 已修

`Evaluator.autoregressive` 结尾原来会再调一次 `teacher_forced()`，只是为了把
`ppl/qa_loss/reconstruction_loss` 塞进同一个 `val/autoregressive/*` 命名空间，
代价是**每次 AR 评测完整多跑一遍验证集**（667 个 batch ≈ 53 s/rank）。
`teacher_forced_every` 和 `autoregressive_every` 还是两个独立周期，这笔开销经常白花。

现在 `Evaluator.autoregressive(..., include_teacher_metrics=False)` 关掉这次重复；
`scripts/test.py` 显式传 `False` 并把 AR 结果作为 headline 先打印；
`scripts/train.py` 在同一个 step 上 TF 已经跑过时传 `False`，否则才补一次。
AR 结果另外镜像到 `val/primary/*`。

### 4.2 同一 context 的多个 QA 合并成一个 batch 做 prefill —— 已实现

数据按 context 聚合，所以一个 context 的所有 QA 天然共享同一份 memory KV cache，
本来就该一次 prefill 完。新增 `MetaLoRA.generate_answers_with_prefix(...,
group_by_context=True, max_rows_per_group=N)`：

- 按 `qa_context_indices` 分组，同组用 `_select_cache` 把 memory cache 复制成 `[group, ...]`；
- 组内 question 左对齐，但**每行用各自的 `position_ids`**，所以某行的真实 question token
  落在和逐行解码完全相同的绝对位置上，左侧的 padding 列在 prefill 和每一步 decode 都被 mask 掉；
- 逐行记录生成结果，每行自己遇 EOS 就停（其余行继续），最后统一 padding。

`generate_answer_with_prefix` 保留为 `group_by_context=False` 的薄封装，
所以老路径仍然可用、可用于对拍。

**实测**（`step-20000.pt`，60 个 SQuAD dev context / 505 个 QA，greedy max_new=32）：

| 路径 | 耗时 | official F1 | official EM |
| --- | --- | --- | --- |
| 逐行（原路径） | 162.1 s | 0.3034 | 0.1861 |
| 分组批量 | **48.9 s（3.3×）** | 0.2980 | 0.1782 |

配对 bootstrap（n=505）：F1 Δ = −0.0054，95% CI [−0.0156, +0.0028] —— **不显著**；
EM Δ = −0.0079，CI [−0.0158, −0.0020]，只有 4/505 行变差。
生成文本 90.9% 逐字节相同。

**为什么不是 100% 相同**：同一个问题下所有 key 的位置、mask 都对齐了，剩下的差异来自
SDPA 在不同 batch size 下选到不同 kernel / 不同归约顺序，在 top-2 logit 接近时被 greedy
argmax 放大。这不是逻辑差异，无法通过改代码消除（除非强制同一 kernel）。

> 顺带修掉一个我自己引入的 bug：增量 decode 的 `attention_mask` 最初用 `torch.ones`，
> 这会把 prefill 阶段被 mask 掉的 question padding key 重新暴露出来。
> 现在 decode 的 mask 是 `[memory(全1), question(按 mask), generated(全1)]`，
> 相同条件下文本一致率从 66.5% 提到 90.9%，F1 差异从不显著为负变成不显著。

## 5. gradient accumulation —— 已整体切除

### 原来的问题

`Accelerator(gradient_accumulation_steps=1)` 与训练循环里手写的累积逻辑并存，四个后果：
① Accelerate 不进入累积模式 → 不会用 `no_sync()`，DDP 每个 micro-batch 都 all-reduce，
GA=4 时通信量 ×4；② `step` 先自增再判 `(step + 1) % GA == 0`，窗口错位成
`{1}, {2,3}, {4,5}…`，**循环结束时不满一个窗口的梯度永远不会被 step**；
③ 绕过 `accelerator.clip_grad_norm_`；④ 早期 bf16 梯度范数不准
（这条已由 `model.trainable_dtype: float32` + `src/dtypes.py` 解决）。

### 现在的状态

**gradient accumulation 机制已整体移除**，每个 DataLoader batch 一次 optimizer step：

- `TrainingConfig` 删掉 `grad_accumulation` 字段，`validate()` 里对应检查也删掉；
- `configs/{default,qwen-1.7b,qwen-4b,qwen-8b}/train.yaml` 里的 `grad_accumulation` 键删掉；
- `scripts/train.py` 的 `total_steps` 变成 `max(1, effective_loader_len * epochs)`；
- 训练循环改成无条件

  ```python
  if accelerator is not None:
      accelerator.backward(total)
  else:
      total.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
  optimizer.step()
  scheduler.step()
  optimizer.zero_grad(set_to_none=True)
  ```

  不再有 `/ cfg.training.grad_accumulation`，也不再有 `%` 门槛。

这样 `total_steps`、scheduler、wandb 的 step 三者严格一致，也不存在被丢弃的尾部梯度。
将来若真要累积，正确的接入方式是 `Accelerator(gradient_accumulation_steps=N)`
+ `with accelerator.accumulate(model)` + `accelerator.backward(loss)`，
而不是在循环里自己数步子。

## 6. `config.py:185` 把 `max_context_tokens` 硬卡在 2048

```python
if not 0 < d.max_context_tokens <= 2048:
    raise ValueError("data.max_context_tokens must be in (0, 2048]")
```

### 根因：这个上限其实是被 decoder 的位置表卡住的，不是 backbone

- backbone 完全不受限：Qwen3-1.7B 的 `max_position_embeddings = 40960`，RoPE 可以外推。
- 真正卡住的是 `MemoryDecoder`（`src/MemoryDecoder.py:30,39,55-56`）：

  ```python
  def __init__(self, qwen_hidden_size, decoder_hidden_size, num_heads, ffn_ratio=2,
               max_context_tokens=2048):
      self.position = nn.Embedding(max_context_tokens, decoder_hidden_size)
  ...
  positions = torch.arange(context_length, device=...)
  query = self.position(positions)[None].expand(...)
  ```

  `MetaLoRA` 把 `max_context_tokens=cfg.data.max_context_tokens` 直接传给了 28 个 decoder
  （`src/model.py` 的 `load_model`），所以**配置里的数据过滤阈值同时就是 decoder 位置表的大小**。
  `context_length > max_context_tokens` 时 `nn.Embedding` 直接越界报错。
- 次要原因：`build_block_causal_mask` 物化 `[B,1,T,T]`（L=8192 时 B=1 就是 128 MB/次），
  以及这个字段还兼任数据集过滤阈值、并参与数据集缓存 fingerprint。

### 现象

`filter_long_context: true` 会把所有超过 2048 token 的 context 从训练**和验证**里删掉。
SQuAD dev 恰好全部 ≤ 2048（实测 2067/2067 保留，长度缓存里 p90=433、max=2048），
所以这个限制在 SQuAD 上看不出来；但 qasper / narrativeqa / ms_marco 的长样本会被静默丢弃，
而 ICL baseline 可以用到 8192 token —— 两者根本不在同一个输入预算下。

### 结论：不改，这个上限是有意的成本控制

把 `max_context_tokens` 定在 2048 同时达到两件事：① 把长 context 从训练/验证里剔除，
省算力；② `MemoryDecoder` 的位置表 `nn.Embedding(2048, 256)` 不再变大 ——
28 层 × 2048 × 256 ≈ 15M 参数，加上 fp32 AdamW 的动量就是 ~180 MB，确实值得省。
这两个目的是**同一个决策**，所以「解耦成两个字段」并不必要，`validate()` 里的硬上限
正好充当这个策略的唯一真源。

需要补的只有两点可观测性：

1. **确认 SQuAD 上没有副作用**：实测 `aggregated/squad/validation.jsonl` 的 2067 个 context
   **全部 ≤ 2048 token，一个都没被丢**，所以 §2 的 SQuAD 对比不受影响。
   受影响的只有 qasper / narrativeqa / ms_marco 这类长 context 数据集。
2. **把「静默丢弃」变成可观测 —— 已落地**：`AggregatedContextDataset._build_records`
   原来只是 `continue`，现在会打印一行统计：

   ```text
   Loaded 2 contexts from 1 file(s) (dropped: 1 longer than 128 tokens, 1 without a usable QA pair, 1 with an empty context)
   ```

如果将来要放宽，注意它同时会改 MD 参数量、数据集缓存 fingerprint（Arrow + sortish 缓存重建）、
以及 `build_block_causal_mask` 的 `[B,1,T,T]` 开销，所以应该一次只动一个变量。

## 7. prefix pass 白算 LM head（已修）+ 死代码清理

### 是什么

`load_model` 用的是 `AutoModelForCausalLM`，所以 `self.qwen` 是
`Qwen3ForCausalLM = Qwen3Model + lm_head`（一个 `hidden 2048 → vocab 151936` 的线性层，
因为 `tie_word_embeddings=True` 与 embedding 共享权重）。
`transformers` 里 `logits_to_keep` 默认是 `0`，而 `slice(-0, None)` 就是整段，
所以 `encode_context_prefix` 里的 `self.qwen(...)` 会把**每一个 context 位置**都过一遍词表头，
吐出 `[B, L+8, 151936]` 的 logits —— 而这个 pass 只读 `out.hidden_states` 和
`out.past_key_values`，**一个 logits 都用不到**。

### 修法

新增 `MetaLoRA._transformer_body`：从 `self.qwen` 往下剥掉任何带 `lm_head` 的包装
（PEFT 装了的话 transformer 会多一层），`encode_context_prefix` 改用它；
`forward_qa_with_prefix` 和 generation 仍然用 `self.qwen`，因为它们要 `.logits`。

### 实测（L=2048, B=1，`step-20000.pt` 同配置）

| | `self.qwen` | `_transformer_body` |
| --- | --- | --- |
| logits | `(1, 2056, 151936)` = **596 MB** | 无 |
| prefix 前向耗时 | 0.47 s | **0.13 s（3.6×）** |
| hidden_states / past_key_values | — | **逐位相同** |

显存：独立进程实测同一 loss（都只对最后隐状态求导）下 **8.28 GiB → 7.70 GiB（−0.58 GiB/行）**，
与 596 MB 的 logits 张量吻合。直接读取分配量也确认：带词表头时 forward 结束瞬间多占 596 MB。

> 说明：`verify_lmhead_skip.py` 里再用 `max_memory_allocated` 做 A/B 会低估这个差值，
> 因为对比第二个变体时第一个变体的 hidden states / logits 还被引用着，把峰值抬高了。
> 以「独立进程的 8.28 → 7.70 GiB」和「直接读 logits 分配量 596 MB」为准。

### 死代码

`_encode_context`（27 行）在 `forward` 改走 `encode_context_prefix` +
`forward_qa_with_prefix` 之后就没有任何调用点了，已整段删除。


<a id="readme-target-format-history"></a>

## 原 README：target-format 修复记录

来源：整理前 `30b897a` 的 README，第 138–185、343–378 行。以下数值与示例保留原样，不代表重新验证。

## Answer-target format and metric semantics

Three issues were found after the first training runs. They are documented here because
two of them silently corrupted the training target instead of raising an error, and the
third made the reported score look better than the model actually was.

### Fixed: question padding used to sit between the question and the answer

`collate_context_records` right-padded questions to the longest question in the batch,
and the model then concatenated `[question, answer]`. For every QA row whose question was
shorter than the batch maximum, the first answer token was therefore predicted from the
hidden state of a *padding* position, whose continuation-mask row contains only a
self-edge. Training and teacher-forced evaluation both supervised the first answer token
there, while `generate_answer_with_prefix` strips the padding before decoding, so the
supervised position and the inference position were not the same one.

The effect is directly measurable through `Evaluator.teacher_forced` on a trained
checkpoint, over 200 SQuAD-dev contexts (1598 QA rows). With the legacy collate, letting
8 contexts share a batch instead of 1 dropped F1 from 0.2907 to 0.2503 and
first-answer-token accuracy from 0.082 to 0.011; with the fix the same comparison gives
0.3860 vs 0.3705 and 0.325 vs 0.296. The small residual change comes from context
padding shifting the absolute positions, not from the question padding. Keeping one row
fixed and only changing its batch neighbour flipped its first predicted token from `in`
(`infinite`) to `No` under the legacy collate.

`data.question_padding_side` now defaults to `left`, which keeps the real question tokens
adjacent to the answer, and `generate_answer_with_prefix` selects question tokens through
`question_mask` instead of slicing a prefix. `build_block_causal_mask` and
`build_continuation_mask` already index questions through `question_mask`, so they are
correct for either padding side. Set `data.question_padding_side: right` only to
reproduce the old runs.

### Fixed: `append_eos` could overwrite the last answer token

`collate_context_records` wrote EOS at `valid`, the first padding slot. That requires a
free slot beyond the longest answer, but `padding=True` pads to the longest answer *in
the batch*, so for the row(s) with `valid == width` the `elif` branch overwrote the final
answer token with EOS instead of appending it. When every answer in a batch had the same
length this destroyed the entire answer:

```text
question_padding_side=right, eos_mode=overwrite  ->  answer_ids [[151645], [151645]]
question_padding_side=left,  eos_mode=append     ->  answer_ids [[32214, 151645], [80185, 151645]]
```

`data.eos_mode` now defaults to `append`, which allocates one extra column and writes EOS
after the last real answer token. `overwrite` remains available to reproduce old runs.

### Regression check for the two target-format fixes

```python
from transformers import AutoTokenizer
from src.data import ContextRecord, QARecord, collate_fn

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
rows = [
    ContextRecord("Alpha beta gamma delta epsilon zeta eta theta.",
                  (QARecord("Which Greek letter is third?", "gamma", "x", ""),), "x", ""),
    ContextRecord("One two three four five six seven eight nine ten.",
                  (QARecord("What comes after six?", "seven", "x", ""),), "x", ""),
]
kwargs = dict(max_context_tokens=128, max_question_tokens=32, max_answer_tokens=16,
              append_eos=True, use_chat_template=False,
              chat_template_enable_thinking=False, qa_per_context=2, sample_qa=False)
fixed = collate_fn(rows, tokenizer, question_padding_side="left", eos_mode="append", **kwargs)
legacy = collate_fn(rows, tokenizer, question_padding_side="right", eos_mode="overwrite", **kwargs)
assert fixed["answer_ids"].tolist() == [[32214, 151645], [80185, 151645]]
assert legacy["answer_ids"].tolist() == [[151645], [151645]]   # answer destroyed by the old bug
assert (fixed["labels"] == fixed["answer_ids"]).all()
```

The fixed layout must also keep the question's real tokens immediately before the answer.
The first row below has the longer question and therefore no question padding; the second
row is left padded:

```python
qmask = fixed["question_mask"]
assert qmask[0].all() and qmask[0].sum() == 6
assert qmask[1].tolist() == [False, True, True, True, True, True]
# real question tokens are last in every row, so the answer starts right after them
for row in qmask:
    assert row[-int(row.sum()):].all() and not row[:-int(row.sum())].any()
```


## 原 README：早期方法概述

以下是历史原文；其 PEFT、pooling 和损失概述不再作为当前方法说明。当前说明见[方法文档](../guides/METHOD.md)。

# Qwen MetaLoRA Context Memory

This project trains a single Qwen backbone with ordinary PEFT MetaLoRA (static LoRA).
Qwen itself encodes the context (there is no second memory encoder). Per-layer pooled
Qwen context states are combined with a global learnable memory-token bank, and the resulting memory is
inserted into the Qwen sequence as `[context, memory, question, answer]`. MetaLoRA is
implemented as ordinary PEFT LoRA (`lora_A/lora_B`) on the selected Qwen projections.
Every Qwen layer has an independent memory decoder. It is trained either to reconstruct the
original context input embeddings, or -- the default -- to predict the context tokens
themselves through a shared vocabulary head, so that the memory has to make the context
recoverable as text rather than as a point in embedding space. The objective is
`qa_weight * answer_only_causal_CE + reconstruction_weight * auxiliary_memory_loss`; see
"Auxiliary memory objectives".


<a id="readme-dtype-history"></a>

## 原 README：dtype 测量记录

以下保留原测量文字；其中 bfloat16 精度表述需更正：显式 fraction 为 7 位，含隐含首位的有效精度为 8 位，1.0 上方相邻数间隔为 2^-7。原数值表没有在本次重算。

### Why

bfloat16 carries 8 mantissa bits, so a value near 1.0 has a relative resolution of about
2^-8. AdamW's step is `p.add_(m_hat / (sqrt(v_hat) + eps), alpha=-lr)`, whose magnitude is
roughly `lr` = 1e-4, below that resolution. The update rounds away. Measured on a tiny
Qwen3 build with all 59 trainable tensors and one AdamW step at `lr=1e-4`: **43/59
parameters changed in bfloat16, 59/59 in float32**. `.tmp_analysis/verify_dtype.py`
reproduces this.

The end-to-end effect, from two arms trained for 6000 steps on the same SQuAD subset with
the same seed and only `model.trainable_dtype` changed (300 SQuAD-dev examples, paired
bootstrap 95% CI):

| metric | bfloat16 | float32 | delta |
| --- | --- | --- | --- |
| autoregressive F1 | 0.2863 | **0.3497** | **+0.0635** [+0.021, +0.105] |
| autoregressive EM | 0.1700 | **0.2267** | **+0.0567** [+0.017, +0.097] |
| teacher-forced F1 | 0.3683 | **0.4364** | **+0.0681** [+0.038, +0.098] |
| teacher-forced EM | 0.1200 | **0.1700** | **+0.0500** [+0.020, +0.080] |

Parameter precision and the auxiliary objective interact: with float32 parameters the
`mse_cosine` reconstruction loss is worth keeping (removing it cost 0.050 autoregressive
F1, CI [−0.092, −0.007]), whereas under bfloat16 it had looked like dead weight. Re-check
any "drop the reconstruction term" conclusion against a float32 run.

