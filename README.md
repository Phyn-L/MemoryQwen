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

### Running on another machine

The shipped configs point at the machine the code was developed on, but no path is baked in:
every location is written as `${VAR:-default}`, so a cluster with different directories
overrides it through the environment instead of editing tracked files -- editing them is what
made `git pull` conflict. `utils.config.expand_env` does the substitution, and
`${VAR}` without a `:-` fallback raises rather than silently expanding to an empty path.

| variable | replaces | cloud default |
| --- | --- | --- |
| `MODEL_ROOT` | the HF hub directory in `model.name_or_path` | `/data/lz/hf_cache/hub` |
| `DATA_ROOT` | `data.root` | `/data/lz/contexts/aggregated` |
| `WANDB_MODE` | `logging.wandb_mode` (use `offline` on a machine without internet) | `online` |
| `CUDA_VISIBLE_DEVICES` | which cards are visible, and therefore how many workers `scripts/train.sh` starts | unset (all) |
| `NUM_PROCESSES` | the worker count, overriding the count derived from the visible cards | derived |
| `CONFIG` | the config every entry-point script uses | `configs/qwen-1.7b/train.yaml` |

Put them in `scripts/env.local.sh`, which is gitignored and sourced by `train.sh`, `test.sh`
and `test_icl_baseline.sh` when present:

```bash
# scripts/env.local.sh -- one machine's settings, never committed
export MODEL_ROOT=/home/lijie/proj2/xmu/lz
export DATA_ROOT=/home/lijie/proj2/xmu/lz/aggregated
export WANDB_MODE=offline
```

`tests/test_config_env.py` checks that the shipped configs still resolve to the cloud defaults
with an empty environment, and to the H200 paths above when those variables are set.

`scripts/train.sh` starts one worker per visible GPU, sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and **fails** if several processes are
requested while `accelerate` is missing. It no longer hardcodes a device list or a process
count: an 8-card node used to run on the 4 cards the script named, and if `accelerate` was not
installed the run quietly continued on a single card.

`.[train]` installs `peft`, but whether `peft` is present no longer decides which LoRA
implementation runs: `model.use_peft` does, and it defaults to `false`. See
[LoRA backend](#lora-backend).

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
template; it is disabled by default. `data.question_padding_side` (default `left`) and
`data.eos_mode` (default `append`) control the answer-target format and are described in
"Answer-target format and metric semantics" below; both defaults are the corrected
behaviour and the legacy values exist only for reproducing older runs.
`model.trainable_dtype` (default `float32`) sets the dtype of the trainable parameters
while `model.torch_dtype` stays the backbone dtype; see "Two dtypes" below.
Empty evaluation JSONL files are accepted. The
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
src/model.py  src/losses.py  src/MemoryDecoder.py  src/dtypes.py  src/data.py
src/dataset_cache.py  utils/optimizer.py  utils/scheduler.py  utils/checkpoint.py
utils/config.py  utils/ddp.py
```

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

### Teacher-forced scores are a diagnostic, not the headline

`Evaluator.teacher_forced` feeds the answer tokens as *inputs* (teacher forcing), so
every answer position after the first can be produced by continuing the prefix it
already attends to. Its `em`/`f1`/`rouge_l`/`precision` therefore reward lexical continuation more
than retrieval, and can even come out below the autoregressive score. Predictions from a
trained checkpoint:

```text
gold 'apoplectic stroke' -> teacher-forced 'Nooplelectic'
gold 'Deabolis'          -> teacher-forced 'Nobal'
gold 'β-defensins'       -> teacher-forced 'V-defensin'
```

The word tails are often right while the entity is wrong. Both evaluators now also report
`first_token_em`, the only answer position that must be produced from memory alone:

- teacher forced: the argmax at the first answer position equals the gold first answer token;
- autoregressive: the first generated token equals the first token of the gold answer.

Read the **autoregressive** numbers as the headline result and teacher-forced
`first_token_em` as the retrieval diagnostic. A large teacher-forced/autoregressive gap
means the memory is not supplying the answer even when teacher-forced F1 looks
respectable. `scripts/test.py` prints the autoregressive result first and labels the
teacher-forced one as a diagnostic, and `Evaluator.autoregressive(...,
include_teacher_metrics=False)` skips the extra teacher-forced pass when the caller
already runs it.

Because the autoregressive number is the only one that is comparable with an ICL
baseline, training must actually measure it: `evaluation.autoregressive_every` controls how
often it runs, and `evaluation.max_new_tokens` is 32 to match
`scripts/test_icl_baseline.py --squad-max-new-tokens 32`. Setting `autoregressive_every` to
a huge number is how the `pgw1382s` run ended up with no trustworthy score at all.

Both evaluation modes are logged into their own W&B section, with identical metric keys so
the panels line up:

| section | contents |
| --- | --- |
| `val_teacher_forced/*` | `em` `f1` `rouge_l` `precision` `first_token_em` (lexical-continuation flavoured — a diagnostic) plus the loss scalars `loss` `qa_loss` `ppl` `reconstruction_loss` |
| `val_autoregressive/*` | the same five answer-quality metrics, produced without feeding the gold answer back — the headline numbers |

W&B groups panels by the first path component, so the section has to lead the key. The two
prefixes are the `TEACHER_FORCED_SECTION` / `AUTOREGRESSIVE_SECTION` constants at the top of
`scripts/train.py`; this replaces the earlier `val/teacher_forced/*` + `val/autoregressive/*`
layout, which put everything in one `val` section, and the `val/primary/*` mirror of the
autoregressive numbers, whose name made the headline result look like it was called
"primary". Loss scalars stay under the teacher-forced section even when the autoregressive
pass computed them internally, and `tests/test_eval_logging.py` pins the whole layout.

### Metric semantics

Answer-quality metrics use the official SQuAD normalizer -- lowercase, punctuation
stripped, articles dropped -- because that is what the published ICL baseline numbers use,
and the two must be the same number for a comparison to mean anything. There is exactly one
implementation (`src/metrics.py`) and one reduction (`best_reference_metrics`), and
`tests/test_metrics.py` asserts that `src.icl_baseline.example_metrics` and the evaluator
agree key for key on every key in `src.metrics.METRIC_KEYS`:

| key | definition |
| --- | --- |
| `em` | normalized prediction equals a gold answer exactly |
| `f1` | token-overlap F1 against the best gold answer |
| `rouge_l` | LCS F1 against the best gold answer |
| `precision` | fraction of predicted tokens present in the reference (formerly, and wrongly, called `bleu`) |

Both evaluators additionally report `first_token_em`: the first answer token must be
produced from memory alone, so it is the retrieval signal that `em`/`f1` are not.

Both evaluators reduce over *every* gold answer with a metric-wise max, and
`src/data.py` now carries all of them through the dataset cache (`QARecord.answers`,
`dataset_cache.CACHE_VERSION = 4`; the older schema kept only the first answer, which made
the evaluator see one reference where the baseline sees up to six -- of the 16498 answered
QA pairs in `aggregated/squad/validation.jsonl`, 12728 have 3 references, 2092 have 5 and
1384 have 4). `first_token_em` uses the same rule: matching any annotator's first token
counts as a hit. `tests/test_references.py` covers the end-to-end path
(jsonl -> dataset -> HF cache -> reload), which is where a missed `CACHE_VERSION` bump would
show up.

Both harnesses read the same file, and it is the only supported one:
`scripts/test_icl_baseline.py` defaults to `<data.root>/squad/validation.jsonl` and reads it
through `src.icl_baseline.iter_examples`, which understands the aggregated context schema and
**raises** on the old one-question-per-line layout instead of yielding an empty evaluation
set. `tests/test_icl_data.py` asserts the parity against the training split on the real files.

### The SQuAD evaluation set, in detail

`<data.root>/squad/*.jsonl` is not "SQuAD dev" in the sense the papers mean. It is the
**v1.1 and v2.0 sets concatenated per paragraph**, and that composition is visible in every
number measured from it:

| | validation | train |
| --- | --- | --- |
| contexts | 2067 | 19030 |
| QA rows | 22443 | 217918 |
| ...tagged `*-v1.1` | 10570 | 87599 |
| ...tagged `*-v2.0` | 11873 | 130319 |
| rows with no gold answer (dropped by both harnesses) | 5945 | 43498 |
| answered rows -- what a metric is averaged over | 16498 | 174420 |
| **distinct `(context, question)`** | **10531** | **87406** |
| answered rows that repeat another row | 5967 (36%) | 87014 (50%) |
| distinct questions answerable in *both* versions | 5915 | 86761 |
| distinct questions whose own repeat rows disagree | 17 | 79 |

What follows from that:

- **The row count is not the sample size.** Both harnesses average over rows, so the 5915
  validation questions present in *both* versions carry roughly twice the weight of the 4616
  that only v1.1 answers -- v2.0 marks those impossible, so they contribute a single row each.
  Comparisons *between two of our own runs* are unaffected, because the weighting is identical
  on both sides. An absolute number quoted next to a published SQuAD figure is not apples to
  apples: the file mixes two annotation vintages and over-weights the overlap.
- **The disagreements are not v1.1-versus-v2.0 annotation drift.** All 17 validation and all
  79 train disagreements are between rows of the *same* version -- the aggregation simply kept
  the same question twice. `what is one name used to refer to the jurisdiction of NCT of ...`
  appears twice as `train-v1.1`, once with `New Delhi` and once with `Delhi`. Several train
  answers are outright fragments (`Buddh` for `Buddhism`, `m and E`, `yptian Se`), which looks
  like a character-offset bug in the aggregation's answer extraction rather than annotation
  noise. Training samples QA pairs per context, so duplicated questions also skew which
  questions get trained on.
- **The 5945 unanswerable rows are v2.0's**, and both harnesses drop them -- `filter_no_qa`
  for training, `load_jsonl` for the baseline -- so the two score the same rows. Nothing
  currently trains or scores SQuAD 2.0's unanswerable task.
- `<data.root>/squad/test.jsonl` is **0 bytes**, so squad has no test split; use validation.

Three ways to clean this up, in increasing scope. None is a scoring change, so none is applied
here -- the current file is the one every recorded run used:

1. de-duplicate by `(context, question)` at load time, keeping the first row: 10531 rows, so
   the row count becomes the sample size; where the duplicate rows disagree the retained
   answer is arbitrary, which is why this is the least satisfying option;
2. keep only `*-v1.1` rows: 10570 validation rows, i.e. exactly SQuAD v1.1 dev and comparable
   with published numbers -- and the same filter on the train file, which is then what the
   model is trained on;
3. fix the aggregation that produced these files: merge by `(context, question)` and reconcile
   `answer_starts`/`answers`. Only this option also addresses the fragment answers, and it
   belongs in whichever script built the aggregated tree.

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

## Two dtypes: bfloat16 backbone, float32 trainable parameters

The frozen Qwen backbone stays in its checkpoint dtype (bfloat16), while `memory_tokens`,
`lora_A`/`lora_B` and the 28 reconstruction decoders are kept in float32. This is not one
global dtype but two dtypes that meet at explicit boundaries.

`model.trainable_dtype` selects it (`float32` by default; set `bfloat16` to reproduce the
earlier runs). `MetaLoRA.__init__` still casts the whole composite model to the backbone
dtype and then calls `set_trainable_dtype(trainable_dtype)`, which matches on parameter
names so it works for both the PEFT path and the `StaticLoRALinear` fallback.

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
measurements in `IMPROVEMENTS.md` were taken with, and because the choice must not depend
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
(default 256) caps how many QA rows *each rank* generates — the global number of decoded
rows is `autoregressive_max_qa * world_size`. Leaving that cap out of a 4-way run is what
produced the 10-minute hang described above.

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
- **One shared vocabulary head, not one per layer.** `head: Linear(D, vocab_size)` is
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

`configs/qwen-1.7b/train.yaml`, `configs/qwen-4b/train.yaml` and
`configs/qwen-8b/train.yaml` all select `context_lm`. Changing the single
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
