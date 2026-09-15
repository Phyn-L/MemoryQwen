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
cached under `<checkpoint.output_dir>/dataset_cache/<split>-<fingerprint>/records.jsonl`.
Reading the source JSONL files and tokenizing every context for the length filter is by
far the most expensive part of startup, and it happens in every process of an
`accelerate launch`; this cache skips it on later runs. `sortish_lengths.json` only
caches the much cheaper length pass that runs after the dataset is already built, so it
does not avoid that work. Set `data.cache_dataset: false` to always rebuild.
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
