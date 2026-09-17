#!/usr/bin/env bash
# Evaluate one checkpoint on the FULL SQuAD validation split, separately for the v1.1 and the
# v2.0 question sets, reporting EM / F1 / ROUGE_L (plus the retrieval diagnostic
# first_token_em). Autoregressive is the headline number; teacher-forced is printed next to it
# as a diagnostic, because it feeds the gold answer prefix back in.
#
# Why a script instead of `scripts/test.py` twice by hand:
#   * `aggregated/squad/validation.jsonl` carries both versions and marks every QA pair with
#     `split: validation-v1.1` / `validation-v2.0` (22443 pairs: 10570 v1.1, 11873 v2.0, of
#     which 5945 are the v2.0 unanswerable ones). The script writes one filtered file per
#     subset under a scratch root, so the same weights are scored on exactly those rows;
#   * the eval config is taken from the checkpoint itself (a training checkpoint records the
#     config that produced it), so the reader switches / memory_length / head_mode the weights
#     were trained with are the ones used to load them -- no guessing which YAML goes with which
#     run. Pass CONFIG=... to override that;
#   * nothing caps the split: `--max-samples` is deliberately not passed, so all 2067 contexts
#     and every QA row of the subset are scored (SAMPLE_CAP=... exists only for a smoke run).
#
# Usage (on the 4090):
#   CHECKPOINT=outputs/Qwen1.7B_20260917_213648.pt bash scripts/eval_squad_v1v2.sh
#   CHECKPOINT=... SAMPLE_CAP=8 bash scripts/eval_squad_v1v2.sh      # quick plumbing check
#   CHECKPOINT=... DRYRUN=1 bash scripts/eval_squad_v1v2.sh          # print the plan only
#   CHECKPOINT=... CONFIG=configs/qwen-1.7b/train.yaml bash ...      # override the derived config
#   CHECKPOINT=... CUDA_VISIBLE_DEVICES=1 bash ...
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

CHECKPOINT="${CHECKPOINT:-outputs/Qwen1.7B_20260917_213648.pt}"
WORK="${WORK:-outputs/eval_squad_v1v2}"
SUBSETS="${SUBSETS:-v1 v2 v2all}"
SAMPLE_CAP="${SAMPLE_CAP:-}"
DRYRUN="${DRYRUN:-0}"
# Four ranks by default: each one gets a shard of the validation loader and the evaluator
# all-reduces the accumulators, so the metrics are the same as a single-process run of the
# same rows (verified: NUM_PROCESSES=1 and 4 agree on a 64-row sample).
NUM_PROCESSES="${NUM_PROCESSES:-4}"
# Evaluation batch size. Empty = whatever the checkpoint's config used (training.batch_size).
# The eval is no_grad, so raising it is a memory/time tradeoff you can measure, not a
# correctness one: the metrics are batch-invariant (verified 8 vs 32 on the same rows).
BATCH_SIZE="${BATCH_SIZE:-}"
QA_BATCH_SIZE="${QA_BATCH_SIZE:-}"
CONFIG="${CONFIG:-}"
MACHINE="${MACHINE:-4090}"
export MACHINE
export WANDB_MODE="${WANDB_MODE:-offline}"

# Card selection. An explicit CUDA_VISIBLE_DEVICES is respected as-is; otherwise every visible
# card stays visible, which is what `accelerate launch --num_processes N` needs -- pinning it to
# "0" here made a 4-rank launch silently run one rank on GPU 0. NUM_PROCESSES is also clamped to
# the visible cards, with a warning, so a typo cannot quietly cost 3/4 of the machine.
VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-}"
if [ -z "$VISIBLE_GPUS" ]; then
  VISIBLE_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
  VISIBLE_GPUS=$(awk -F, '{print NF}' <<<"$VISIBLE_GPUS")
fi
case "$VISIBLE_GPUS" in ''|*[!0-9]*) VISIBLE_GPUS=1 ;; esac
[ "$VISIBLE_GPUS" -ge 1 ] || VISIBLE_GPUS=1
if [ "$NUM_PROCESSES" -gt "$VISIBLE_GPUS" ]; then
  echo "eval_squad_v1v2.sh: NUM_PROCESSES=$NUM_PROCESSES but only $VISIBLE_GPUS GPU(s) visible" \
       "(CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}); running $VISIBLE_GPUS." >&2
  NUM_PROCESSES="$VISIBLE_GPUS"
fi

# Per-machine paths come from utils/machines.py (MACHINE=4090 above); the gitignored env file
# still wins for a one-off override -- same rule as train.sh.
if [ -f "$ROOT/scripts/env.local.sh" ]; then
  # shellcheck source=/dev/null
  . "$ROOT/scripts/env.local.sh"
fi

PYTHON="${PYTHON:-python}"
command -v "$PYTHON" >/dev/null 2>&1 || { echo "eval_squad_v1v2.sh: '$PYTHON' not on PATH" >&2; exit 1; }
# The chain runs `python` again inside scripts/test.sh, so a python that cannot import torch is
# a mistake worth catching here rather than three frames into the split step.
if ! "$PYTHON" -c "import torch" >/dev/null 2>&1; then
  echo "eval_squad_v1v2.sh: '$PYTHON' ($(command -v "$PYTHON")) cannot import torch." >&2
  echo "                  activate the project environment (conda activate shine) or pass" >&2
  echo "                  PYTHON=/path/to/env/bin/python." >&2
  exit 1
fi
[ -f "$CHECKPOINT" ] || {
  echo "eval_squad_v1v2.sh: checkpoint '$CHECKPOINT' not found; pass CHECKPOINT=<path>" >&2
  exit 1
}

mkdir -p "$WORK"
echo "checkpoint : $CHECKPOINT"
echo "scratch    : $WORK"
echo "machine    : $MACHINE (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all $VISIBLE_GPUS>}, NUM_PROCESSES=$NUM_PROCESSES)"
echo "subsets    : $SUBSETS${SAMPLE_CAP:+   (SAMPLE_CAP=$SAMPLE_CAP rows per subset)}${CONFIG:+   (config override: $CONFIG)}"
echo "batch      : ${BATCH_SIZE:-<checkpoint config>} contexts/batch${QA_BATCH_SIZE:+, qa group $QA_BATCH_SIZE}"
echo

# --- 1. one filtered validation file + one eval config per subset ---------------------------
"$PYTHON" - "$CHECKPOINT" "$WORK" "$SUBSETS" "$CONFIG" <<'PY'
"""Split SQuAD validation by its version marker, then write one eval config per subset.

`aggregated/squad/validation.jsonl` marks every QA pair with `split: validation-v1.1` or
`validation-v2.0`, so the two question sets separate exactly and without touching the shared
data tree: each subset gets a scratch root holding `<root>/squad/validation.jsonl`, which is
all `make_context_dataset` reads.

The eval config is the one stored inside the checkpoint, filtered to the fields the current
dataclasses still accept -- a checkpoint from an older revision keeps working instead of
failing on a key that has since been removed, and the dropped keys are reported.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

import torch
import yaml

from utils.config import (
    CheckpointConfig,
    DataConfig,
    EvaluationConfig,
    LoggingConfig,
    MemoryConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from utils.machines import fill_missing, machine_environ, resolve_machine

SECTIONS = {
    "model": ModelConfig, "memory": MemoryConfig, "data": DataConfig,
    "optimizer": OptimizerConfig, "scheduler": SchedulerConfig,
    "evaluation": EvaluationConfig, "training": TrainingConfig,
    "checkpoint": CheckpointConfig, "logging": LoggingConfig,
}

checkpoint = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
subsets = sys.argv[3].split()
config_override = sys.argv[4]

state = torch.load(checkpoint, map_location="cpu", weights_only=False)
state_dict = dict(state.get("model") or {})


def shape_truth(tensors: dict) -> dict:
    """Shape-defining config values read from the checkpoint's own tensors.

    A checkpoint stores the config that produced it, but that is only as trustworthy as the
    file: a `CONFIG=` override can easily describe a *different* run (this is exactly how the
    first user of this script got "size mismatch for memory_tokens" -- the 4090's default
    train.yaml is M=8/ctx=2048 while this checkpoint is M=64/ctx=1024), and an old checkpoint
    may predate a config field. The tensors cannot lie, so they decide.
    """
    truth: dict = {}
    memory = tensors.get("memory_tokens")
    if memory is not None and memory.ndim == 2:
        truth["memory_length"] = int(memory.shape[0])
    position = tensors.get("decoders.0.position.weight")
    if position is not None and position.ndim == 2:
        truth["max_context_tokens"] = int(position.shape[0])
        truth["decoder_hidden_size"] = int(position.shape[1])
    if "context_lm_head.adapter.weight" in tensors:
        truth["head_mode"] = "tied"
    elif "context_lm_head.weight" in tensors:
        truth["head_mode"] = "linear"
    latents = tensors.get("resampler.latents")
    if latents is not None:
        truth["readout_length"] = int(latents.shape[0])
        truth["readout_hidden_size"] = int(latents.shape[1])
    else:
        truth["readout_length"] = 0
    for key, tensor in tensors.items():
        if key.endswith("lora_A") and tensor.ndim == 2:
            truth["lora_rank"] = int(tensor.shape[0])
            break
    return truth


truth = shape_truth(state_dict)
described = f"M={truth.get('memory_length')} ctx={truth.get('max_context_tokens')} head={truth.get('head_mode')}"
if config_override:
    # A caller-pinned config: read it the normal way (this also resolves ${VAR:-default} and
    # the machine table), then dump it back so the per-subset copies share one code path.
    from utils.config import TrainConfig

    raw_config = dataclasses.asdict(TrainConfig.from_file(config_override))
    print(f"[config] base = {config_override} (caller override; the checkpoint says {described})")
else:
    raw_config = {key: value for key, value in dict(state.get("config") or {}).items() if key in SECTIONS}
    if not raw_config:
        raise SystemExit(f"{checkpoint} carries no config; pass CONFIG=<yaml> to pin one")
    print(f"[config] base = the config stored in {checkpoint.name}")

resolved, authoritative = resolve_machine()
overrides = machine_environ(resolved)
if not authoritative:
    overrides = fill_missing(overrides, os.environ)
data_root = (
    overrides.get("DATA_ROOT")
    or os.environ.get("DATA_ROOT")
    or "/data/lz/contexts/aggregated"
)
model_root = (
    overrides.get("MODEL_ROOT")
    or os.environ.get("MODEL_ROOT")
    or "/data/lz/hf_cache/hub"
)
source = Path(data_root) / "squad" / "validation.jsonl"
if not source.is_file():
    raise SystemExit(f"{source} not found; set DATA_ROOT (or MACHINE=...) to the aggregated tree")

# --- split ----------------------------------------------------------------------------------
def keep_pair(subset: str, pair: dict) -> bool:
    if subset == "v1":
        return pair.get("split") == "validation-v1.1"
    if subset == "v2":
        return pair.get("split") == "validation-v2.0" and bool(pair.get("answers"))
    if subset == "v2all":
        return pair.get("split") == "validation-v2.0"
    raise SystemExit(f"unknown subset {subset!r}; use v1 / v2 / v2all")


counts: dict[str, dict[str, int]] = {}
for subset in subsets:
    subset_root = work / subset / "squad"
    subset_root.mkdir(parents=True, exist_ok=True)
    contexts = pairs_kept = 0
    with source.open(encoding="utf-8") as reader, (subset_root / "validation.jsonl").open(
        "w", encoding="utf-8"
    ) as writer:
        for line in reader:
            if not line.strip():
                continue
            row = json.loads(line)
            pairs = [pair for pair in (row.get("qa_pairs") or []) if keep_pair(subset, pair)]
            if not pairs:
                continue
            copy = dict(row)
            copy["qa_pairs"] = pairs
            writer.write(json.dumps(copy, ensure_ascii=False) + "\n")
            contexts += 1
            pairs_kept += len(pairs)
    counts[subset] = {"contexts": contexts, "pairs": pairs_kept}
    (work / f"counts_{subset}.json").write_text(json.dumps(counts[subset]), encoding="utf-8")
    print(
        f"[split]  {subset:<5} contexts={contexts:5} qa_rows={pairs_kept:6} "
        f"-> {subset_root / 'validation.jsonl'}"
    )

# --- one config per subset ------------------------------------------------------------------
def cleaned_sections(config: dict) -> tuple[dict, list[str]]:
    cleaned, dropped = {}, []
    for section, cls in SECTIONS.items():
        values = dict(config.get(section) or {})
        allowed = {field.name for field in dataclasses.fields(cls)}
        for key in list(values):
            if key not in allowed:
                dropped.append(f"{section}.{key}")
                values.pop(key)
        cleaned[section] = values
    return cleaned, dropped


def apply_shape_truth(cleaned: dict, truth: dict) -> list[str]:
    """Force the shape-defining fields to the values the checkpoint's tensors have.

    Everything here changes the *architecture*, so a wrong value is not a style question: it
    either fails the load with a wall of size mismatches or (worse, for fields that keep the
    shapes) evaluates the weights through a different model. `memory_length`,
    `max_context_tokens`, `decoder_hidden_size`, `head_mode`, `readout_length` and `lora_rank`
    are read off the tensors, and any disagreement with the config is reported.
    """
    memory, data = cleaned["memory"], cleaned["data"]
    wanted = (
        (memory, "memory_length"),
        (data, "max_context_tokens"),
        (memory, "decoder_hidden_size"),
        (memory, "head_mode"),
        (memory, "readout_length"),
        (memory, "readout_hidden_size"),
        (cleaned["model"], "lora_rank"),
    )
    corrections = []
    for section, key in wanted:
        value = truth.get(key)
        if value is None or section.get(key) == value:
            continue
        corrections.append(f"{key} {section.get(key)} -> {value}")
        section[key] = value
    return corrections


for subset in subsets:
    cleaned, dropped = cleaned_sections(raw_config)
    corrections = apply_shape_truth(cleaned, truth)
    if corrections:
        print(f"[config] {subset}: corrected from the checkpoint's tensors: " + "; ".join(corrections))
    cleaned["data"]["root"] = str((work / subset).resolve())
    cleaned["data"]["validation_datasets"] = ["squad"]
    cleaned["data"]["test_datasets"] = ["squad"]
    cleaned["data"]["train_datasets"] = ["squad"]           # unused by scripts/test.py
    cleaned["data"]["validation_max_samples"] = None         # the whole subset
    cleaned["data"]["filter_no_qa"] = subset != "v2all"       # v2all keeps the unanswerable rows
    cleaned["logging"]["wandb_mode"] = "offline"
    cleaned["checkpoint"]["output_dir"] = str(work)
    # A checkpoint stores the *resolved* model path of the machine it was trained on (the
    # configs interpolate ${MODEL_ROOT} at load time and the file keeps the result), so a
    # checkpoint trained on one machine has to be re-pointed at this machine's hub directory
    # before it can be evaluated here.
    stored = str(cleaned["model"].get("name_or_path") or "")
    marker = "models--"
    if marker in stored:
        repointed = str(Path(model_root) / stored[stored.index(marker):])
        if repointed != stored:
            cleaned["model"]["name_or_path"] = repointed
            print(f"[config] {subset}: model path re-pointed to {model_root}")
    elif stored and not Path(stored).exists():
        print(f"[config] WARNING: {subset}: model path {stored!r} does not exist here and has no "
              f"'models--' component to re-point; pass CONFIG=... or set MODEL_ROOT")
    path = work / f"config_{subset}.yaml"
    path.write_text(yaml.safe_dump(cleaned, sort_keys=False, allow_unicode=True), encoding="utf-8")
    note = f"   (dropped unsupported keys: {', '.join(dropped)})" if dropped else ""
    print(f"[config] {path}{note}")
PY

echo

# --- 2. evaluate every subset ---------------------------------------------------------------
if [ "$DRYRUN" != "0" ]; then
  for subset in $SUBSETS; do
    echo "would run: MACHINE=$MACHINE CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NUM_PROCESSES=$NUM_PROCESSES \\"
    echo "  bash scripts/test.sh --config $WORK/config_${subset}.yaml \\"
    echo "    --checkpoint $CHECKPOINT --split validation${SAMPLE_CAP:+ --max-samples $SAMPLE_CAP}"
  done
  echo "DRYRUN OK (nothing evaluated)"
  exit 0
fi

for subset in $SUBSETS; do
  log="$WORK/test_${subset}.log"
  echo "=== $subset -> $log ==="
  extra=()
  # No --max-samples unless SAMPLE_CAP asks for one: scoring the full split is the point.
  [ -n "$SAMPLE_CAP" ] && extra=(--max-samples "$SAMPLE_CAP")
  [ -n "$BATCH_SIZE" ] && extra+=(--batch-size "$BATCH_SIZE")
  [ -n "$QA_BATCH_SIZE" ] && extra+=(--qa-batch-size "$QA_BATCH_SIZE")
  MACHINE="$MACHINE" NUM_PROCESSES="$NUM_PROCESSES" CONFIG="$WORK/config_${subset}.yaml" \
    bash scripts/test.sh --checkpoint "$CHECKPOINT" --split validation "${extra[@]}" 2>&1 | tee "$log"
  echo
done

# --- 3. one summary table -------------------------------------------------------------------
"$PYTHON" - "$WORK" "$SUBSETS" "$CHECKPOINT" "$SAMPLE_CAP" <<'PY'
"""Turn the per-subset stdout into one table, plus results.md / results.json."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

work, subsets, checkpoint = Path(sys.argv[1]), sys.argv[2].split(), sys.argv[3]
cap = sys.argv[4] if len(sys.argv) > 4 else ""
KEYS = ("em", "f1", "rouge_l", "first_token_em")
TF_KEYS = ("em", "f1", "rouge_l")
LABELS = {
    "v1": "v1.1 (all answerable)",
    "v2": "v2.0 (answerable only)",
    "v2all": "v2.0 (all; unanswerable scored as misses)",
}

rows = []
for subset in subsets:
    log = work / f"test_{subset}.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    parsed = {"autoregressive": {}, "teacher_forced": {}}
    for line in text.splitlines():
        for section, marker in (
            ("autoregressive", "autoregressive (headline): "),
            ("teacher_forced", "teacher-forced (diagnostic only): "),
        ):
            if marker in line:
                try:
                    parsed[section] = ast.literal_eval(line.split(marker, 1)[1])
                except (ValueError, SyntaxError):
                    pass
    counts = {}
    counts_file = work / f"counts_{subset}.json"
    if counts_file.is_file():
        counts = json.loads(counts_file.read_text(encoding="utf-8"))
    rows.append(
        {
            "subset": subset,
            "label": LABELS.get(subset, subset) + (f" [CAP {cap}]" if cap else ""),
            "contexts": counts.get("contexts"),
            "subset_rows": counts.get("pairs"),
            "sample_cap": int(cap) if cap.isdigit() else None,
            "autoregressive": parsed["autoregressive"],
            "teacher_forced": parsed["teacher_forced"],
        }
    )

if not any(row["autoregressive"] for row in rows):
    print("no metrics parsed -- check the per-subset logs under", work)
    raise SystemExit(1)

width = max(len(row["label"]) for row in rows)
header = (
    f"{'subset':<{width}}  {'sub rows':>8}  {'AR EM':>7} {'AR F1':>7} {'AR R-L':>7} {'AR 1st':>7}"
    f"  |  {'TF EM':>7} {'TF F1':>7} {'TF R-L':>7}"
)
print(header)
print("-" * len(header))


def cell(value) -> str:
    return "    n/a" if value is None else f"{float(value):7.4f}"


for row in rows:
    ar, tf = row["autoregressive"], row["teacher_forced"]
    print(
        f"{row['label']:<{width}}  {row['subset_rows'] or '':>8}  "
        + " ".join(cell(ar.get(key)) for key in KEYS)
        + "  |  "
        + " ".join(cell(tf.get(key)) for key in TF_KEYS)
    )
print()
if cap:
    print(f"NOTE: SAMPLE_CAP={cap} -- the numbers above are the first {cap} contexts of each")
    print("      subset (a plumbing check), not the whole split. Drop SAMPLE_CAP for the real run.")
    print()
print("AR = autoregressive (headline, no gold prefix); TF = teacher-forced (diagnostic).")
print("Quote 'v1.1 (all answerable)' and 'v2.0 (answerable only)' for a version comparison.")
print("The 'v2.0 (all)' row scores the unanswerable questions as misses: this model has no")
print("abstention mechanism, it always emits an answer.")

(work / "results.json").write_text(
    json.dumps({"checkpoint": checkpoint, "subsets": rows}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
lines = [
    f"# SQuAD validation, full split -- `{checkpoint}`",
    "",
    "| subset | contexts | subset QA rows | AR EM | AR F1 | AR ROUGE-L | AR first_token_em | TF EM | TF F1 | TF ROUGE-L |",
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
]
for row in rows:
    ar, tf = row["autoregressive"], row["teacher_forced"]
    lines.append(
        f"| {row['label']} | {row['contexts'] or ''} | {row['subset_rows'] or ''} | "
        + " | ".join(cell(ar.get(key)).strip() for key in KEYS)
        + " | "
        + " | ".join(cell(tf.get(key)).strip() for key in TF_KEYS)
        + " |"
    )
(work / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nwrote {work / 'results.md'} and {work / 'results.json'}")
PY
