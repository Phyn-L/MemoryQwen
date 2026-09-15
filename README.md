# Qwen MetaLoRA Context Memory

This project trains a single Qwen backbone with ordinary PEFT MetaLoRA (static LoRA).
Qwen itself encodes the context (there is no second memory encoder). Per-layer pooled
Qwen context states are combined with a global learnable memory-token bank, and the resulting memory is
inserted into the Qwen sequence as `[context, memory, question, answer]`. MetaLoRA is
implemented as ordinary PEFT LoRA (`lora_A/lora_B`) on the selected Qwen projections.
Every Qwen layer has an independent decoder trained to reconstruct
the original context input embeddings. The objective is
`qa_weight * answer_only_causal_CE + reconstruction_weight * length_normalized_reconstruction`.

The default data root is `/data/lz/contexts/aggregated`. Each dataset item is one context; empty-QA records and contexts
whose tokenized length exceeds 2048 are removed before a sortish length sampler is built.
All valid QA pairs belonging to a context are retained. Training samples at most
`data.qa_per_context` (4 by default) QA pairs randomly on every collate call, while
validation and test expand every QA pair. A context batch of size C therefore has a
separate QA batch of size Q (Q <= C*qa_per_context in training), with
`qa_context_indices` mapping each QA row back to its context row.

```bash
pip install -e '.[train]'
PYTHONPATH=. bash scripts/train.sh
```

Each fresh run is named like `Qwen1.7B_20260915_173045`. This name is used for
the W&B run and checkpoint directory `outputs/<run-name>/`. Reusable dataset and
sortish caches are stored separately by model under `outputs/Qwen1.7B/`, so new
runs do not rebuild the same filtered dataset. When resuming, the existing
checkpoint directory is reused.

Configuration is YAML-only. Complete model-specific configurations are maintained as YAML files:
`configs/qwen-1.7b/train.yaml`, `configs/qwen-4b/train.yaml`, and
`configs/qwen-8b/train.yaml`. Each contains all sections needed for a reproducible
run. Set `data.train_datasets`,
`data.validation_datasets`, and `data.test_datasets` independently, with corresponding
`*_max_samples` limits. Contexts longer than 2048 tokens and records without a usable
QA pair are filtered before the sortish sampler is constructed. Set
`data.use_chat_template: true` to render questions through the tokenizer's chat
template; it is disabled by default. Empty evaluation JSONL files are accepted. The
training entry point only loads train and validation data; test evaluation is handled
by `scripts/test.py`. Training, optimizer, scheduler, evaluation, checkpoint, and
logging settings live in their corresponding top-level config sections.
With `data.cache_sortish_lengths: true` (the default), tokenized sample lengths are
cached as `sortish_lengths.json` under the model cache directory, for example
`outputs/Qwen1.7B/`. The cache is rebuilt automatically when its data,
tokenizer, filtering, or chat-template fingerprint changes.
With `data.cache_dataset: true` (the default), the filtered context records themselves are
cached as a Hugging Face Arrow dataset under
`outputs/QwenXB/dataset_cache/<split>-<fingerprint>/hf_dataset/`.
The first run converts the filtered records to Arrow with `Dataset.save_to_disk()`.
Later runs use `Dataset.load_from_disk()` and avoid JSONL parsing; `sortish_lengths.json`
still caches the cheaper length pass separately. Set `data.cache_dataset: false` to
always rebuild from the source JSONL files.
The dataset and sortish caches are context-level and use bumped format versions, so
old flat-QA caches (including ~998k-row length caches) are ignored automatically.
The model path uses a differentiable context-prefix cache: each context is encoded
once, and its memory prefix is gathered with `qa_context_indices` for all of that
context's QA rows. Evaluation uses `evaluation.qa_batch_size` (4 by default) to
micro-batch QA continuations while accumulating metrics by the actual QA count.
Use the matching `train.yaml` with `--config` when selecting another backbone.

The source layout is:

```text
configs/qwen-1.7b/train.yaml
configs/qwen-4b/train.yaml
configs/qwen-8b/train.yaml
scripts/train.py  scripts/test.py  scripts/train.sh  scripts/test.sh
src/model.py  src/losses.py  src/MemoryDecoder.py  src/data.py
src/dataset_cache.py  utils/optimizer.py  utils/scheduler.py  utils/checkpoint.py
utils/config.py  utils/ddp.py
```

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

Each decoder in `src/MemoryDecoder.py` receives only its own layer's memory
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

## Embedding reconstruction loss

The reconstruction target is the original context **input embedding** sequence,
not the hidden state from a later Qwen layer. For a decoder prediction `\hat{x}`
and target embedding `x`, both with shape `[B, num_layers, L, H]` after
broadcasting across layers, the loss is averaged over valid (non-padding) context
tokens and all Qwen layers. The implemented components are

```text
MSE     = mean_H((\hat{x} - x)^2)
Cosine  = 1 - cosine_similarity(\hat{x}, x)
```

The default configuration uses `reconstruction_loss: mse_cosine` and
`reconstruction_cosine_weight: 0.1`, giving

```text
L_recon = MSE + 0.1 * Cosine
```

This combines coordinate-wise fidelity (including embedding magnitude) with a
directional or semantic alignment term. The coefficient `0.1` is a baseline
hyperparameter rather than a theoretically fixed ratio; the numerical and
gradient scales of the two terms should be reported when comparing variants.
The reconstruction objective is then combined with answer-only causal QA loss as
`qa_weight * L_QA + reconstruction_weight * L_recon`.

The following settings define useful ablations for studying what information the
memory must preserve. Only setting C is implemented by the current
`mse_cosine` loss; settings D and E describe planned alternatives and require
additional loss code.

| Setting | Reconstruction objective | Hypothesis tested |
| --- | --- | --- |
| A | MSE | Is precise, coordinate-wise recovery important? |
| B | Cosine | Is preserving embedding direction alone sufficient? |
| C | MSE + 0.1 Cosine | Current baseline combining magnitude and direction. |
| D | Cosine + norm loss | Should direction and embedding magnitude be constrained separately? |
| E | Cosine + relational loss | Is preserving token-to-token representation structure more important? |

For setting D, a norm term can penalize the difference between `\|\hat{x}\|`
and `\|x\|`. For setting E, a relational term can compare the token-token
similarity matrices of the predicted and target sequences. These alternatives
should be compared using both reconstruction metrics and downstream QA metrics; a
lower embedding loss alone does not establish that the memory is more useful for
question answering.

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
