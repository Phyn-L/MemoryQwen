# 评测协议与指标

当前实现入口：[evaluator](../../src/evaluator.py)、[metrics](../../src/metrics.py)、[data](../../src/data.py)。2026-09-18 整理；下列样例分数、数据集行数是原 README 的历史快照，不是本次重新测量。正式比较时记录数据指纹、过滤规则、checkpoint、解码预算和实际 QA 数。

当前默认 `question_padding_side=left`、`eos_mode=append`；旧模式只用于复现实验。修复过程及回归示例见[历史记录](../history/ENGINEERING.md#readme-target-format-history)。主指标使用自由生成的 AR EM/F1；TF 用作诊断。数据过滤仍会影响 SQuAD v2 无答案行，不能将有答案子集得分解释为完整 SQuAD 2.0 成绩。

运行命令见[运行指南](RUNNING.md)，SQuAD 专用评测异常见[异常记录](../experiments/EVAL_ANOMALIES.md)。

### Teacher-forced scores are a diagnostic, not the headline

`Evaluator.teacher_forced` feeds the answer tokens as *inputs* (teacher forcing), so
every answer position after the first can be produced by continuing the prefix it
already attends to. Its `em`/`f1`/`rouge_l` therefore reward lexical continuation more
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

"Equals" means the same token id *or* the same normalized text, because the same word has
two encodings depending on the leading space: the gold answer tokenized on its own gives
`cat`, while the same word continued after a question is ` cat`. This is not a corner
case: on Qwen3-1.7B **none** of the 55028 gold first tokens in
`aggregated/squad/validation.jsonl` keeps its id under a leading space, so the id
comparison accepts only the bare encoding and every space-encoded generation is a false
miss. Tokens that normalize to nothing (punctuation, specials) still match by id only, so
the looser comparison cannot invent hits; and since the id comparison is tried first, the
change can only *add* hits -- `first_token_em` numbers reported before it are lower bounds,
not a different scale.

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
| `val_teacher_forced/*` | `em` `f1` `rouge_l` `first_token_em` (lexical-continuation flavoured — a diagnostic) plus the loss scalars `loss` `qa_loss` `ppl` `reconstruction_loss` |
| `val_autoregressive/*` | the same four answer-quality metrics, produced without feeding the gold answer back — the headline numbers |

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

There used to be a fourth key, `precision` (unigram precision over the prediction, once
wrongly called `bleu`). It was dropped because it is gameable in the direction this task
already leans: a one-token prediction that occurs in the reference scores 1.0 while a
fully correct longer answer scores below it, and it tracked `f1` almost exactly on the
runs we have. `f1` is the harmonic mean of the same overlap and stays comparable with the
SQuAD literature. The real BLEU still lives in `src/icl_baseline.corpus_bleu`.

Both evaluators additionally report `first_token_em`: the first answer token must be
produced from memory alone, so it is the retrieval signal that `em`/`f1` are not.

Both evaluators reduce over *every* gold answer with a metric-wise max, and
`src/data.py` now carries all of them through the dataset cache (`QARecord.answers`,
`dataset_cache.CACHE_VERSION = 4`; the older schema kept only the first answer, which made
the evaluator see one reference where the baseline sees up to six -- of the 16498 answered
QA pairs in `aggregated/squad/validation.jsonl`, 12728 have 3 references, 2092 have 5 and
1384 have 4). `first_token_em` uses the same rule: matching any annotator's first token
counts as a hit, compared as a token id or as normalized text so the leading-space
encoding of a generated token does not hide a hit. `tests/test_references.py` covers the
end-to-end path
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
