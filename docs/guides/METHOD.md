# 方法与实现

2026-09-18 对照代码 `30b897a` 整理。实现入口：[model](../../src/model.py)、[losses](../../src/losses.py)、[训练目标组合](../../scripts/train.py)、[配置字段](../../utils/config.py)。历史耗时、显存与数值验证保留为原记录，未在本次重测。

Qwen backbone 编码 `[context, memory slots]`，提取逐层 memory 隐状态及仅包含 memory 位置的 KV cache；QA continuation 通过 `qa_context_indices` 复用对应 context 的 memory cache。context KV 不进入 QA continuation。默认使用内置 StaticLoRA；`model.use_peft=true` 才选用 PEFT。

默认 YAML 使用 QA CE + context-LM 辅助目标。开启额外目标时，训练总损失还包括 `ae_lm_weight * L_AE` 与 `distill_weight * L_distill`。这些分支和 readout 开关见 [reader 选项](READER_OPTIONS.md)。下文双项公式描述这两个额外权重为零的基础路径；“默认”需区分 dataclass 默认与所选 YAML（例如 dataclass 的 reconstruction_loss 是 mse_cosine，而随附 train.yaml 选择 context_lm）。

## Two dtypes: bfloat16 backbone, float32 trainable parameters

The frozen Qwen backbone stays in its checkpoint dtype (bfloat16), while `memory_tokens`,
`lora_A`/`lora_B` and the 28 reconstruction decoders are kept in float32. This is not one
global dtype but two dtypes that meet at explicit boundaries.

`model.trainable_dtype` selects it (`float32` by default; set `bfloat16` to reproduce the
earlier runs). `MetaLoRA.__init__` still casts the whole composite model to the backbone
dtype and then calls `set_trainable_dtype(trainable_dtype)`, which matches on parameter
names so it works for both the PEFT path and the `StaticLoRALinear` fallback.

### Why

float32 trainable parameters reduce rounding of small optimizer updates. Historical dtype measurements and paired comparisons are preserved in [the engineering record](../history/ENGINEERING.md#readme-dtype-history); they are evidence for those runs, not a guarantee for every configuration.

### The three casts

`self.dtype` stays the *compute* dtype: every tensor that enters Qwen is cast to it, so
the KV cache and `memory_cache` are always bfloat16. Only the parameter dtype changes.

1. **Memory tokens.** The parameter is float32 and is cast down where it becomes an input:

   ```python
   memory_inputs = self.memory_tokens.to(dtype=self.dtype).unsqueeze(0).expand(B, -1, -1)
   ```

   `.to()` is differentiable, so the gradient comes back in float32 — the same situation
   as a float32 embedding table feeding a bfloat16 network.

2. **LoRA.** `StaticLoRALinear` keeps `base` in the backbone dtype, runs the low-rank
   branch in the adapter dtype with autocast disabled, and casts the delta back before
   adding it:

   ```python
   base_out = self.base(x.to(base_dtype))
   with no_autocast(x.device):
       lora_x = self.dropout(x).to(self.lora_A.dtype)
       delta = self.scaling * (lora_x @ self.lora_A.t() @ self.lora_B.t())
   return base_out + delta.to(base_dtype)
   ```

   This is the same dtype bookkeeping PEFT performs (`previous_dtype = x.dtype` … cast
   back), but PEFT does *not* disable autocast around its LoRA matmuls, so the two
   implementations are equivalent only when autocast is off. See
   [LoRA backend](#lora-backend).

3. **Reconstruction decoders.** `MemoryDecoder` casts `memory_embedding` to its own
   parameter dtype on entry and returns that dtype. `reconstruction_loss` then upcasts
   prediction and target to float32 before reducing, because the context target is the
   backbone's bfloat16 input embedding. `qa_loss` likewise reduces the answer-only cross
   entropy in float32 instead of in the logits' dtype.

### LoRA backend

There are two implementations of the same adapter, selected by `model.use_peft`
(default `false`):

| `model.use_peft` | implementation | autocast around the LoRA matmuls |
| --- | --- | --- |
| `false` (default) | in-repo `StaticLoRALinear` | disabled, so float32 stays float32 |
| `true` | `peft.get_peft_model` | disabled by `disable_autocast_for_peft_lora` |

Both are kept, but the static one is the default because it is the one the float32
measurements in [IMPROVEMENTS](../history/IMPROVEMENTS.md) were taken with, and because the choice must not depend
on whether `peft` happens to be installed — installing the `[train]` extra is not a
statement about numerics. `use_peft: true` without `peft` installed raises instead of
silently falling back.

Regression coverage for the dtype contract lives in `tests/test_lora_dtype.py`: that the
static delta is computed in float32 under bf16 autocast (with a discriminating check that
the autocast path really is worse), that the default backend is static, and that the PEFT
wrapper turns autocast off inside a `lora_A`/`lora_B` layer without changing its result.

### Autocast

`Accelerator(mixed_precision="bf16")` wraps the training forward in `torch.autocast`,
which casts matmuls to bfloat16 for speed regardless of parameter dtype. That is the
standard AMP contract and it is what keeps this cheap: parameters, their `.grad` and
AdamW's moments stay float32 even though most arithmetic runs in bfloat16.
`src/dtypes.py:no_autocast` additionally disables autocast inside the adapter and decoder
bodies, so those small matmuls really execute in float32 at negligible cost.

### Cost

Roughly 8e7 trainable parameters, so float32 parameters plus AdamW moments add about
0.8 GB over the bfloat16 baseline. On a 24 GB card that matters: the earlier runs peaked
near 21.7 GB and three of them died with CUDA OOM. Pair this change with skipping the
reconstruction decoders when `reconstruction_weight: 0`, with `gradient_checkpointing`, or
with a smaller context batch.

Checkpoints stay compatible in both directions because `load_state_dict` casts on copy, so
a bfloat16 `last.pt` loads into float32 parameters and vice versa.

## Context-prefix sharing

The training and evaluation data path has two batch dimensions:

```text
C = number of contexts in the DataLoader batch
Q = number of QA rows represented by those contexts

context_ids:         [C, L]
question_ids:        [Q, QL]
answer_ids:          [Q, AL]
qa_context_indices:  [Q]
```

`qa_context_indices[j]` identifies the context that owns QA row `j`. Training
samples at most `data.qa_per_context` QA pairs per context on every collate call;
validation and test retain every valid QA pair.

`src/model.py` exposes two stages:

```python
prefix = model.encode_context_prefix(context_embeds, context_mask)
output = model.forward_qa_with_prefix(
    prefix, qa_context_indices, question_embeds, question_mask,
    answer_embeds, answer_mask, labels,
)
```

`encode_context_prefix()` runs Qwen once on `[context, memory]` for the C contexts
and returns:

- per-layer memory hidden states `[C, num_layers, M, H]`;
- final memory `[C, M, H]`;
- reconstruction predictions `[C, num_layers, L, H]`;
- a memory-only KV cache for each context;
- the context target and mask used by reconstruction loss.

`forward_qa_with_prefix()` selects prefix rows with differentiable
`index_select`, then runs only the `[question, answer]` continuation for Q QA rows.
The original `forward()` remains as a compatibility wrapper that constructs an
identity QA mapping and routes through the same two-stage implementation.

### Differentiable KV cache

The prefix cache is an in-memory Hugging Face `DynamicCache`, not a disk cache and
not a cache shared across optimizer steps. It is created and consumed within one
forward/backward pass. Only the memory part of the prefix KV is retained, because
the attention mask prevents question and answer tokens from reading context tokens
directly.

The selected cache is built from tensor indexing without `detach()`, `.data`,
`numpy()`, or `torch.no_grad()` in the training path. Consequently, gradients from
multiple QA rows accumulate into the same context prefix:

```text
QA_1 loss ─┐
QA_2 loss ─┼─> shared memory KV ─> context prefix ─> context / memory / LoRA
QA_3 loss ─┘
```

The cache itself is treated as read-only. Each QA chunk receives a new cache whose
keys and values are differentiable indexed views of the original prefix cache;
question and answer KV are then appended only to that chunk-local cache. This avoids
in-place batch selection and prevents one QA chunk from contaminating another.

Question and answer position ids remain absolute positions from the original joint
sequence: question positions start at `context_length + memory_length`, rather than
at the shorter memory-only cache length. This preserves Qwen rotary-position
semantics.

### Three execution paths

- **Training:** one prefix encode per local context batch, followed by the sampled
  QA continuation rows. The prefix graph remains attached until the combined loss
  backward pass.
- **Teacher-forced validation:** one prefix encode per context batch, then all QA
  rows are processed in `evaluation.qa_batch_size` chunks. No QA is dropped or
  truncated.
- **Autoregressive evaluation/test:** one prefix encode per context batch; each QA
  gets an independent question/answer continuation cache and incremental token
  generation. Prefix computation is shared, while generated-token state remains
  QA-specific.

The DDP boundary is the local DataLoader shard. Every rank builds and consumes
prefixes only for its own contexts; no prefix hidden state or KV tensor is gathered
across ranks. Only scalar loss/metric reductions are suitable for distributed
aggregation.

### Validation must run on every rank

`accelerator.prepare` shards the validation loader, so restricting the evaluation to the
main process is wrong twice over:

1. the metric would cover only the main rank's shard — with `world_size` ranks and
   `validation_max_samples: 2000`, that is 500 strided contexts rather than the whole
   split;
2. the other ranks would run ahead into the next DDP collective and block there for the
   entire evaluation. Anything longer than the NCCL watchdog timeout (10 minutes)
   aborts the whole job with `Watchdog caught collective operation timeout ...
   OpType=BROADCAST`. That is how a run dies at exactly the step where an
   autoregressive evaluation starts.

`Evaluator.teacher_forced` and `Evaluator.autoregressive` therefore all-reduce their
accumulated sums and counts (`_distributed_sum`) and must be called by every rank;
`scripts/train.py` keeps `is_main` only for `run.log`. Decoding is orders of magnitude
more expensive than a teacher-forced forward, so `evaluation.autoregressive_max_qa`
(default 1024) caps how many QA rows the *whole* evaluation generates. The budget is global:
it is split into contiguous windows of the validation split's row order, so the scored rows —
and therefore the metric's noise — do not depend on how many ranks the run uses. (It used to
be a per-rank cap, which made an 8-GPU and a 4-GPU run score different question sets; the
global decoded count was `autoregressive_max_qa * world_size`.) Leaving the cap out of a
4-way run entirely is what produced the 10-minute hang described above.

In embedding-regression mode, each decoder in `src/MemoryDecoder.py` receives only its own layer's memory
embedding (`[B, M, H]`). It projects memory and positional queries into a
configurable bottleneck of width `memory.decoder_hidden_size`, applies
cross-attention and an FFN with expansion ratio `memory.decoder_ffn_ratio`, and
projects the result back to `H`. The decoder therefore reconstructs the complete
context embedding sequence (`[B, L, H]`), and `src/model.py` stacks these outputs
as `[B, num_layers, L, H]`. The Qwen-1.7B configuration uses a width of 256, an
FFN ratio of 2, and 8 attention heads while retaining one independent decoder per
Qwen layer.
Here `memory.memory_length` is the number of memory tokens `M` (8 by default),
while `qwen_hidden_size` is each token's Qwen feature width `H` (2048 for
Qwen3-1.7B); these are independent dimensions.

## Auxiliary memory objectives

The QA loss is a weak teacher for the memory on its own. A context contributes only a
handful of answer tokens, so a memory that captures the context's topic but none of its
entities still reduces the QA loss a little, and the answer tokens themselves are the
only positions that ever have to be right. An auxiliary objective over the *context* is
what forces content into the memory. `memory.reconstruction_loss` selects which one is
used:

| `memory.reconstruction_loss` | Objective | Target | Typical value |
| --- | --- | --- | --- |
| `mse`, `cosine`, `mse_cosine` | embedding regression | context **input embeddings** `[C, num_layers, L, H]` | ~0.08 |
| `context_lm` | context next-token prediction | context **token ids**, cross entropy | ~12 nats/token |

Both are scaled by `memory.reconstruction_weight` and reported under `reconstruction_loss`
in W&B. The shape of the objective is unchanged:

```text
L_QA    = mean over Q QA-row losses
L_aux   = mean over C contexts of the selected auxiliary loss
L_total = qa_weight * L_QA + reconstruction_weight * L_aux
```

The number of QA pairs belonging to a context changes its QA supervision, but does not
multiply that context's auxiliary loss.

### Embedding reconstruction (`mse_cosine`)

The reconstruction target is the original context **input embedding** sequence, not the
hidden state from a later Qwen layer. For a decoder prediction `\hat{x}` and target
embedding `x`, both with shape `[C, num_layers, L, H]`, the loss is first reduced per
context over valid (non-padding) context tokens and all Qwen layers, then averaged over
the C contexts:

```text
L_recon = MSE + reconstruction_cosine_weight * (1 - cosine_similarity(\hat{x}, x))
```

This combines coordinate-wise fidelity (including embedding magnitude) with a directional
term. The coefficient is a baseline hyperparameter rather than a theoretically fixed
ratio; the numerical and gradient scales of the two terms should be reported when
comparing variants. The following ablations are useful for studying what the memory must
preserve:

| Setting | Reconstruction objective | Hypothesis tested |
| --- | --- | --- |
| A | MSE | Is precise, coordinate-wise recovery important? |
| B | Cosine | Is preserving embedding direction alone sufficient? |
| C | MSE + cosine | Baseline combining magnitude and direction. |
| D | Cosine + norm loss | Should direction and embedding magnitude be constrained separately? |
| E | Cosine + relational loss | Is preserving token-to-token representation structure more important? |

### Why the embedding regression is a weak objective

Three reasons, in increasing order of importance.

1. **The target is a deterministic function of the token id.** The decoder is asked to
   invert the embedding lookup from `M` memory vectors. That is a much harder problem than
   predicting the token, and it is not the problem the QA path needs solved.
2. **MSE has a degenerate optimum that carries no token identity.** The minimiser of
   `E||\hat{x} - x||^2` is the conditional mean of the embedding distribution, a vector
   that may be close to no vocabulary entry at all. A model that predicts something like
   "a plausible average token embedding" reduces the loss while storing nothing usable.
   Cosine similarity does not fix this; it only removes the magnitude term.
3. **It is on a different scale from the QA loss.** The embedding regression saturates
   around 0.08 while the QA cross entropy is 2-18, so `reconstruction_weight` is hard to
   reason about and the auxiliary term is either negligible or dominant depending on a
   constant nobody measured. Empirically the loss stops moving early: in the
   `co5t73u7` and `pgw1382s` runs it was flat at 0.0755 and 0.3723 respectively from
   step 2000 to the end of the run, and raising `reconstruction_cosine_weight` from 0.1
   to 0.5 left the QA loss plateaued at 2.63 instead of improving to 1.79.

### Context next-token prediction (`context_lm`)

The auxiliary target becomes the context **token id** at the sampled positions, scored
with cross entropy through a shared vocabulary head:

```text
L_ctxlm = mean over layers of  mean over valid positions of  CE(head(hidden_t), context_token_t)
```

The query for the target position `t` sits at `t - 1`, so producing token `t` requires the
memory. Everything else about the decoder is unchanged.

#### Design choices that matter

- **Queries attend to the memory only.** The decoder has no self-attention over context
  positions, so it cannot look at token `t - 1`, `t - 2`, ... Adding causal self-attention
  to the decoder would let it behave as a small standalone language model of the context
  and the memory would receive almost no gradient. Keeping the cross-attention to memory as
  the only input channel is what makes this a compression objective rather than a second
  language model.
- **One shared vocabulary head, not one per layer.** The following parameter counts describe `head_mode=linear`; `tied` uses a frozen embedding plus a trainable adapter (see [reader options](READER_OPTIONS.md)). `head: Linear(D, vocab_size)` is
  shared by all `num_layers` decoders, so it costs `D * vocab_size` ~= 39M parameters.
  One head per layer would cost `num_layers * D * vocab_size` ~= 1.09B. Sharing also
  puts every layer's hidden state in a common space before unembedding, which keeps the
  per-layer losses comparable.
- **The head reads the bottleneck width `D`, not `H`.** Applying a head to the
  `output_projection` result would mean `H * vocab_size` per row, eight times the cost at
  `D = 256`, `H = 2048`: roughly 64% of the backbone forward pass versus about 8%. For the
  same reason `MemoryDecoder` is built with `reconstruct_embeddings=False` in this mode and
  does not create `output_projection` at all, which would otherwise be 28 * (D * H + H)
  parameters with zero gradient and a matching AdamW state allocation.
- **Positions are sampled, not exhaustive.** `memory.context_lm_positions` (default 256,
  `0` = every position) bounds how many context positions are scored per context per step,
  sampled without replacement and covering the context uniformly. The head runs on
  `num_layers * context_lm_positions` rows, so this is the knob that controls the extra
  cost. Scoring all 2048 positions of every context through 28 decoders would apply the
  head to ~57k rows. The default 256 is not a compromise for the common case: the training
  contexts have mean length 177 and median 108 tokens, so the budget is only reached and
  subsampling only happens for the long tail, and the cap is what keeps a 2048-token
  context from dominating a step.
- **Logits are never materialised for all layers at once, and each chunk is
  checkpointed.** `context_lm_loss` walks the layers one at a time and chunks rows
  (`max_logits_rows=256`). Chunking alone is not enough: autograd keeps every chunk's
  logits alive for the backward pass, which is `num_layers * max_logits_rows * vocab_size`
  floats, about 4.4 GB for Qwen3-1.7B (28 x 256 x 151936 x 4 bytes), and that dominated
  the measured peak (10.8 GiB before the fix). Wrapping each chunk in
  `torch.utils.checkpoint` recomputes the head in the backward pass instead; it costs one
  extra head forward (~8% of the backbone) and brings the peak back to 7.1 GiB.
- **Both terms are now nats per token.** The context-LM loss starts near
  `ln(vocab_size)` ~= 11.9, and the memory-only QA loss starts in the same range, so
  `qa_weight: 1.0` with `reconstruction_weight: 1.0` is a meaningful starting point
  instead of an arbitrary ratio between two incomparable scales. The absolute value is
  *not* comparable with the embedding-regression value; only its trend is.

#### Cost

Measured with the Qwen3-1.7B config on one 4090, 120 steps at batch size 1 on SQuAD train
contexts (`.tmp_analysis/exlm/train_probe.py`, a real `Qwen3-1.7B` forward and backward,
not a toy model):

| objective | per step | peak allocated |
| --- | --- | --- |
| `mse_cosine` | 0.518 s | 6.88 GiB |
| `context_lm`, 256 positions | 0.603 s | 7.11 GiB |

So the switch costs about **16% throughput and 0.23 GiB**, with no change to the data path
or the checkpoint format. The head accounts for 91.7M - 67.6M = 24.1M extra trainable
parameters (`256 * 151936` ~= 39M for the head, minus the 14.7M `output_projection`
parameters that this mode no longer creates). In the same probe the context-LM loss fell
from 12.10 to 8.98 nats/token over 120 steps while the QA loss fell from 5.93 to 2.74; the
probe is far too short to show a QA benefit, it only establishes that the objective
optimises and what it costs.

#### Switching

```yaml
memory:
  reconstruction_loss: context_lm   # or mse_cosine / mse / cosine
  context_lm_positions: 256         # 0 scores every context position
```

`configs/4090/qwen-1.7b/baseline/train_baseline.yaml`, `configs/4090/qwen-4b/baseline/train_baseline.yaml` and
`configs/4090/qwen-8b/baseline/train_baseline.yaml` all select `context_lm`. Changing the single
`reconstruction_loss` line back to `mse_cosine` restores the previous behaviour; the
checkpoint format is identical either way apart from the head, and `load_state_dict`
runs with `strict=False`, so an old checkpoint still loads (its `output_projection`
tensors are simply unused when the new mode is active, and vice versa).

#### Verifying that the memory is actually being used

A cross entropy over the vocabulary can in principle be lowered by the learned positional
query alone (a per-position unigram prior), which would leave the memory doing nothing.
Before trusting the objective, check that the loss depends on the memory:

```python
# same sampled positions, decoder input replaced by zeros
zero_terms = model.context_lm_terms(
    context_ids, context_mask, torch.zeros_like(prefix.layer_memory),
    cfg.memory.context_lm_positions, targets=(terms.positions, terms.mask),
)
context_lm_loss(terms.hidden, terms.labels, model.context_lm_head, terms.mask)      # real memory
context_lm_loss(zero_terms.hidden, terms.labels, model.context_lm_head, terms.mask)  # no memory
```

Passing the same `targets` keeps the position sample fixed so the two numbers are
comparable. A large positive gap means the memory is carrying the context. A small gap
means the objective has collapsed onto the positional prior and the memory is not being
trained by it.

## Verification and numerical invariants

The prefix-sharing path was checked with a tiny Qwen3 configuration using two
contexts and four QA rows. The shared-prefix path produced:

```text
prefix.memory:         [2, 3, 32]
prefix.reconstruction: [2, 2, 5, 32]
QA logits:              [4, 6, 101]
labels:                 [4, 6]
maximum logit difference
vs. joint [context,memory,question,answer] forward: 8.94e-08
```

The same check confirmed non-empty gradients for `memory_tokens`, LoRA
parameters, and reconstruction decoder parameters after a combined QA plus
reconstruction backward pass. Repeated QA rows share the prefix through
differentiable indexing, so their gradients accumulate at the owning context.

Repository-level checks:

```bash
python -m compileall -q src utils scripts
git diff --check
```

Full-scale training and million-row cache rebuilding are intentionally not part of
the focused verification path. Before production training, repeat the numerical
equivalence and gradient checks with the target Qwen checkpoint and dtype.
