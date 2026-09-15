# Qwen MetaLoRA Context Memory

This project trains a single Qwen backbone with ordinary PEFT MetaLoRA (static LoRA).
Qwen itself encodes the context (there is no second memory encoder). Per-layer pooled
Qwen context states are combined with a global learnable memory-token bank, and the resulting memory is
inserted into the Qwen sequence as `[context, memory, question, answer]`. MetaLoRA is
implemented as ordinary PEFT LoRA (`lora_A/lora_B`) on the selected Qwen projections.
Every Qwen layer has an independent decoder trained to reconstruct
the original context input embeddings. The objective is
`qa_weight * answer_only_causal_CE + reconstruction_weight * length_normalized_reconstruction`.

The default data root is `/data/lz/contexts/aggregated`. Empty-QA records and contexts
whose tokenized length exceeds 2048 are removed before a sortish length sampler is built.

```bash
pip install -e '.[train]'
PYTHONPATH=. bash scripts/train.sh
```

Each fresh run is named like `Qwen1.7B_20260915_173045`. This name is used for
the W&B run, the checkpoint directory, and the dataset/sortish caches under
`outputs/<run-name>/`. When resuming, the existing checkpoint directory is reused.

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
cached as `sortish_lengths.json` under the configured checkpoint output directory,
for example `outputs/qwen-1.7b/`. The cache is rebuilt automatically when its data,
tokenizer, filtering, or chat-template fingerprint changes.
With `data.cache_dataset: true` (the default), the filtered records themselves are
cached as a Hugging Face Arrow dataset under `<checkpoint.output_dir>/dataset_cache/<split>-<fingerprint>/hf_dataset/`.
The first run converts the filtered records to Arrow with `Dataset.save_to_disk()`.
Later runs use `Dataset.load_from_disk()` and avoid JSONL parsing; `sortish_lengths.json`
still caches the cheaper length pass separately. Set `data.cache_dataset: false` to
always rebuild from the source JSONL files.
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
