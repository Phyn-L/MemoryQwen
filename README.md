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

`configs/default.json` is dependency-light; `configs/default.yaml` is provided for
readability when PyYAML is installed. Set `data.train_datasets`,
`data.validation_datasets`, and `data.test_datasets` independently, with corresponding
`*_max_samples` limits. Contexts longer than 2048 tokens and records without a usable
QA pair are filtered before the sortish sampler is constructed. Set
`data.use_chat_template: true` to render questions through the tokenizer's chat
template; it is disabled by default. Empty evaluation JSONL files are accepted. The
training entry point only loads train and validation data; test evaluation is handled
by `scripts/test.py`. Training, optimizer, scheduler, evaluation, checkpoint, and
logging settings live in their corresponding top-level config sections. Legacy flat
training/checkpoint/logging fields remain readable for compatibility.
With `data.cache_sortish_lengths: true` (the default), tokenized sample lengths are
cached as `sortish_lengths.json` under the configured checkpoint output directory,
for example `outputs/qwen-1.7b/`. The cache is rebuilt automatically when its data,
tokenizer, filtering, or chat-template fingerprint changes.
The local Qwen paths mirror `/data/lz/mkv/configs/qwen1.7b.yaml`, `qwen4b.yaml`, and
`qwen8b.yaml`; use the corresponding JSON override with `--config` after merging it
with `configs/default.json` if selecting another backbone.

The source layout is:

```text
configs/qwen-1.7b/train.yaml
configs/qwen-4b/train.yaml
configs/qwen-8b/train.yaml
scripts/train.py  scripts/test.py  scripts/train.sh  scripts/test.sh
src/model.py  src/losses.py  src/MemoryDecoder.py  src/data.py
utils/optimizer.py  utils/scheduler.py  utils/checkpoint.py
utils/config.py  utils/ddp.py
```

Each decoder in `src/MemoryDecoder.py` receives only its own layer's memory
embedding (`[B, M, H]`) and reconstructs the complete context embedding sequence
(`[B, L, H]`). `src/model.py` stacks these outputs as `[B, num_layers, L, H]`.
