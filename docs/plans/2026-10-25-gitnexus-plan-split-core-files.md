# GitNexus Engineering Plan

> Task: Split oversized core files without changing model, data, training, evaluation, or configuration behavior.
> Evidence verified at commit `3979ac11ca347e0c410d2493509da0f5cb5fd2d4`; GitNexus index refreshed this session with `node .gitnexus/run.cjs analyze --index-only --pdg` (GitNexus 1.6.12, PDG enabled).
> Evidence provenance schema 2; global dirty digest `0a9c85780067d9afcd0764f307b60891e3cee927ee11eaeb5ec7826d10fd82cd`; cited-path manifest 22 sorted entries; exact generated plan path excluded.

## 1. Objective

Reduce the maintenance cost of the five oversized core files while preserving public imports, tensor shapes/dtypes, cache semantics, task schedules, checkpoint/resume behavior, YAML resolution, and evaluation metrics. The refactor is staged so every step leaves the repository runnable and reviewable.

Acceptance criteria:

- `src/model.py`, `src/data.py`, `scripts/train.py`, `utils/config.py`, and `src/evaluator.py` become thin composition/facade modules; each extracted module owns one cohesive concern.
- Existing import paths and CLI commands continue to work during the refactor; no compatibility aliases are added beyond preserving currently used public symbols.
- CPU tests, ruff, and the CI command pass after every stage.
- Model outputs/losses/gradients, cache bytes, schedule decisions, evaluation metrics, and checkpoint round-trips are behaviorally equivalent on the existing fixtures.
- No new model architecture, data rule, configuration field, or training policy is introduced.

## 2. Current Behaviour

[verified] `ContextMemoryModel` spans `src/model.py:134-1335` and combines LoRA injection/dtype management, transformer-body resolution, context encoding, memory resampling, QA/reconstruction forward paths, generation, and conversation state/cache management. `ContextPrefix`, `ConversationState`, and `ContextMemoryOutput` are the state carriers shared by those paths (`src/model.py:13-44`).

[verified] `src/data.py` spans 1248 lines and combines JSONL parsing/conversation normalization, serial and process-pool filtering, cache length computation, dataset loading, sortish sampling, prompt construction, and batch collation. `filter_rows_batch` is the single filtering implementation used by both cache building and loader paths (`src/data.py:37-95,224-251`).

[verified] `scripts/train.py::main` spans `306-1063` and performs CLI/config setup, model/device/Accelerate setup, cache preparation, loader creation, resume planning, optimizer/scheduler construction, checkpoint management, training stages, evaluation, logging, and final save.

[verified] `utils/config.py::TrainConfig` spans `387-637`; its loader combines YAML `_base_` inheritance, environment expansion, machine path resolution, grouped data flattening, dataclass construction, legacy-field rejection, serialization, and validation.

[verified] `src/evaluator.py::Evaluator` spans `114-605` and combines distributed evaluation helpers, autoregressive QA/reconstruction evaluation, row-window selection, metric aggregation, and shared QA/Recon sampling.

## 3. Relevant Architecture

[verified] The current dependency direction is already mostly one-way: `scripts/train.py` composes `src.cache_build`, `src.data`, `src.evaluator`, `src.model`, `src.pipeline`, and `utils.*`; `src.data` imports `utils.ddp` and prompt helpers; `src.model` imports `src.resampler`, `src.losses`, and configuration only inside `load_model`.

[verified] Existing cohesive modules provide extraction targets: `src/resampler.py` owns memory/question resampler layers; `src/losses.py` owns objective reduction; `src/pipeline.py` owns loader/window construction; `utils/checkpoint.py` owns persistence.

[inferred] The safest design is to extract by state/side-effect boundary, then keep the current files as facades. Moving symbols first and changing algorithms in the same commit would mix structural and behavioral risk.

## 4. GitNexus Findings

[graph] `context(ContextMemoryModel)` reports 17 direct importing files, including `scripts/train.py`, `scripts/test.py`, `src/evaluator.py`, GPU diagnostics, and model-focused tests. `impact(ContextMemoryModel, upstream, maxDepth=3, summaryOnly=true)` reports 8 direct dependents and MEDIUM risk; all direct consumers must be rechecked after moves.

[graph] `context(filter_rows_batch)` reports direct callers `_filter_row_batch_in_worker`, `filter_row_chunks`, and cache tests. `impact(filter_rows_batch, upstream, maxDepth=3)` reports HIGH risk, 2 direct dependents, 4 affected processes, and 3 affected modules. The cache builder and loader must continue to call one canonical filter implementation.

[graph] `context(main, scripts/train.py)` shows the training entrypoint calls cache creation, model loading, pipeline construction, optimizer/scheduler setup, reporting, and checkpoint/evaluation paths. Its PDG has control-dependent regions for task fallback sampling (`scripts/train.py:102-116`), resume/output path setup (`281-303`), optimizer-step/logging (`817-883`), and checkpoint/evaluation branches (`910+`).

[graph] `context(TrainConfig)` reports 13 direct importers and one depth-2 dependent. `impact(TrainConfig, upstream, maxDepth=3)` reports MEDIUM risk. Config extraction must preserve every caller's `from_file`, `from_dict`, `to_dict`, and `validate` behavior.

[graph] `context(Evaluator)` reports direct consumers `scripts/train.py`, `scripts/test.py`, and evaluation tests. Its three public methods are `autoregressive`, `recon_teacher_forced`, and `compare_reader_tasks`.

[verified] Existing regression tests are behavior-oriented and map well to the extraction boundaries: model writer/checkpointing (`tests/test_context_writer.py`), generation/conversation (`tests/test_generation.py`, `tests/test_chat_template_boundaries.py`), positions (`tests/test_positions.py`), cache/filter equivalence (`tests/test_build_cache.py`), resume (`tests/test_resume.py`), schedule (`tests/test_train_schedule.py`), and config/entrypoints (`tests/test_entrypoints.py`).

## 5. Statement-Level PDG Findings

[graph] PDG controls for `scripts/train.py` show task masks are sampled independently and then repaired when both tasks are absent (`scripts/train.py:102-116`). Extraction must keep this operation inside one step-level helper and preserve the random generator and fallback probability.

[graph] PDG controls for `scripts/train.py::main` show optimizer stepping, scheduler stepping, gradient clipping, metric accumulation, and progress updates are one guarded unit (`scripts/train.py:817-883`). The extracted training loop must not move scheduler stepping into Accelerate preparation or change the optimizer-step boundary.

[graph] PDG controls for the training entrypoint show checkpoint/evaluation branches depend on `boundary`, `run`, `log_step`, `stage`, and `is_main` (`scripts/train.py:910+`). Preserve rank gating, stage-specific managers, final-step evaluation, and checkpoint-before-evaluation ordering.

[graph] PDG controls for `src/data.py` show `filter_row_chunks` selects a process pool only when workers are enabled and otherwise calls `filter_rows_batch` directly (`src/data.py:224-251`). Preserve bounded pending futures, returned length sidecars, and serial/parallel byte equivalence.

[graph] PDG flows for `utils/config.py::TrainConfig.from_dict` show the expanded `values` object feeds model, memory, data, optimizer, checkpoint, logging, evaluation, training, and final dataclass construction (`utils/config.py:432-488`). Extraction must keep one expansion/normalization pass and preserve unknown/moved-field errors.

[verified] Model source shows LoRA is enabled for context/question hidden encoding and disabled for decoder/cache passes (`src/model.py:336-351,525-559,623-692`). It also shows generation and conversation mutate/copy `past_key_values`, while `ContextPrefix` is the shared handoff object (`src/model.py:27-34,849-1135`). These state transitions are hard extraction boundaries.

## 6. Proposed Changes

### 6.1 Model decomposition

- `src/model.py`: retain public dataclasses, constants, `MetaLoRA`, `ContextMemoryModel` facade, `recon_task_token_ids`, and `load_model`; reduce the class to composition/delegation and stable public methods.
- New `src/model_backbone.py`: move `MetaLoRA`, transformer-body resolution, dtype/trainable-parameter helpers, causal-mask/position helpers, and LoRA context manager. It must not import training or evaluation code.
- New `src/model_memory.py`: move context hidden capture, gradient-checkpoint toggling, `encode_context_prefix`, and independent-question hidden encoding. It returns the existing `ContextPrefix` fields without changing cache creation conditions.
- New `src/model_readout.py`: move `_readout_embeddings` and readout-mode-specific assembly. Reuse `src.resampler` and preserve `memory_question`, `learned_prefix`, and `off` behavior.
- New `src/model_tasks.py`: move QA forward, Recon forward, `recon_teacher_forced`, and output/loss assembly. It may call `src.losses`, but must not own generation or conversation state.
- New `src/model_generation.py`: move batched/row generation and prefix cache copying. Preserve `logits_to_keep=1`, greedy decoding, EOS padding, position numbering, and immutable prefix cache semantics.
- New `src/model_conversation.py`: move `start_conversation`, recompression, `append_conversation_turn`, and conversation cache replay. It owns `ConversationState` transitions and remains batch-size-1.
- Keep delegation methods on `ContextMemoryModel` until all callers/tests migrate; do not change caller-facing signatures in this refactor.

### 6.2 Data decomposition

- `src/data.py`: retain re-exported public records and a small facade for existing imports.
- New `src/data_records.py`: move `ConversationMessage`, `ConversationTurn`, `ContextRecord`, JSONL row parsing, turn derivation, and conversation rendering.
- New `src/data_filter.py`: move worker globals, `filter_options`, `filter_rows_batch`, bounded process-map logic, and `filter_row_chunks`. This becomes the only filtering implementation used by `src/cache_build.py` and `ConversationDataset`.
- New `src/data_sampling.py`: move context length measurement, `SortishSampler`, deterministic turn sampling, prompt trimming, and position helpers that are not model-specific.
- New `src/data_collate.py`: move `collate_context_records`, encoding helpers, task packing, and batch tensor assembly. Keep the batch dictionary schema unchanged.
- New `src/data_dataset.py`: move `ConversationDataset`, cache metadata/freshness handling, and row read/write helpers. Keep cache fingerprint inputs and sidecar paths unchanged.
- `src/cache_build.py` remains the orchestration layer and imports filtering primitives from `src.data_filter`.

### 6.3 Training decomposition

- `scripts/train.py`: retain CLI parsing and a short `main` that wires components.
- New `src/training_schedule.py` or `utils/training_schedule.py`: move pure helpers `resume_plan`, `optimizer_steps`, `micro_batch_skip`, `training_stage`, `sample_task_masks`, and `should_evaluate`; preserve their current signatures so existing tests remain valid.
- New `src/training_setup.py`: move config/device/Accelerate setup, cache preflight, dataset/loader construction, optimizer/scheduler construction, and checkpoint manager selection. Return a typed run context rather than a large mutable local namespace.
- New `src/training_loop.py`: move optimizer-step accumulation, stage/epoch iteration, gradient clipping, scheduler stepping, metric payload construction, and progress updates. It must receive dependencies explicitly and keep the current optimizer-step ordering.
- New `src/training_evaluation.py`: move validation loader/window creation, evaluator calls, metric reporting, best-checkpoint selection, and final-step evaluation.
- New `src/training_resume.py`: move resume/init-from handling, stage selection, rank-count warning, and output directory setup. Reuse `utils.checkpoint.CheckpointManager`.

### 6.4 Configuration decomposition

- `utils/config.py`: retain public dataclasses and `TrainConfig` import path, but delegate parsing/normalization to small modules.
- New `utils/config_loader.py`: move `_deep_merge`, `_load_yaml_with_base`, environment expansion, and grouped-data flattening.
- New `utils/config_schema.py`: move the dataclass definitions and field-level defaults; avoid changing serialized keys.
- New `utils/config_validation.py`: move `TrainConfig.validate` checks into section validators called in the same order, preserving error text where tests depend on it.
- `TrainConfig.from_file/from_dict/to_dict` remain the stable orchestration API and must continue to round-trip checkpoints and machine overrides.

### 6.5 Evaluation decomposition

- `src/evaluator.py`: retain `Evaluator` facade and public methods.
- New `src/evaluation_distributed.py`: move distributed rank/world helpers, reductions, row-window calculations, and progress creation.
- New `src/evaluation_qa.py`: move autoregressive QA evaluation and answer-token matching.
- New `src/evaluation_recon.py`: move teacher-forced reconstruction aggregation.
- New `src/evaluation_sampling.py`: move shared deterministic selection and `compare_reader_tasks` orchestration. Preserve QA/Recon sample reuse and per-dataset aggregation.

## 7. Implementation Sequence

1. **Baseline and contracts.** Record current `pytest -q`, `ruff check .`, and focused model/cache/config tests. Fix no behavior. Add import-level contract tests only where a moved public symbol would otherwise be ambiguous.
2. **Extract pure/shared helpers first.** Move model position/mask/LoRA helpers, data records/filter helpers, training schedule helpers, and config loader helpers. Keep forwarding imports in original modules. Run focused tests after each sub-step.
3. **Extract model memory/readout/task paths.** Introduce the model submodules and delegate from `ContextMemoryModel`. Validate tensor shapes, dtype transitions, LoRA enabled state restoration, gradient checkpoint cleanup, and `ContextPrefix` fields.
4. **Extract generation/conversation.** Move generation and conversation code only after model memory/task paths are stable. Validate prefix cache copying, greedy decode outputs, EOS handling, recompression thresholds, history replay, and batch-size-1 errors.
5. **Extract data dataset/collate/sampling paths.** Keep `src.data` re-exports. Run serial/parallel cache equivalence, sidecar, dataset selection, turn sampling, and collate shape tests.
6. **Extract evaluator.** Split distributed helpers, QA, Recon, and shared sampling while keeping `Evaluator` methods as the public façade. Validate metric keys, per-dataset counts, rank reductions, and shared selection.
7. **Extract training setup/loop/resume.** Move code in dependency order: setup → resume → evaluation → loop. Keep `scripts/train.py` as the CLI boundary. Validate two-stage transitions, mix fallback sampling, gradient accumulation, scheduler cadence, final evaluation, checkpoint-before-evaluation, and rank gating.
8. **Extract config schema/validation.** Move dataclass definitions and validators after all consumers are stable. Verify `_base_` inheritance, environment/machine precedence, unknown/moved fields, and checkpoint config round-trip.
9. **Remove only dead local definitions.** Once graph/text search confirms no internal imports depend on old private names, delete duplicate implementations while retaining intentionally public facade imports. Do not add broad compatibility layers.
10. **Final graph and regression gate.** Run `node .gitnexus/run.cjs analyze --index-only --pdg`, then `detect_changes({scope:"all"})`; require non-partial, non-truncated output. Run full pytest, ruff, and CI-equivalent commands.

## 8. Test Strategy

Existing tests to preserve/update:

- `tests/test_context_writer.py`: context layer capture, checkpointing, LoRA state restoration, Recon task embedding, configuration validation.
- `tests/test_generation.py` and `tests/test_chat_template_boundaries.py`: batched/row generation, cache isolation, conversation replay/recompression, EOS and template boundaries.
- `tests/test_positions.py`: shared position numbering and training/generation alignment.
- `tests/test_build_cache.py`: serial/parallel filter byte identity, freshness fingerprints, lengths sidecars, dataset cache behavior.
- `tests/test_resume.py` and `tests/test_train_schedule.py`: optimizer-step schedule, final evaluation, resume stage boundaries, scheduler ownership, logging payloads.
- `tests/test_entrypoints.py`: config round-trip, machine relocation, checkpoint evaluation, launcher behavior.

Add focused contract tests only where extraction creates a new boundary:

- Importing each facade exposes the existing public names and dataclass fields.
- `ContextMemoryModel.state_dict()` keys and trainable parameter grouping are unchanged before/after delegation.
- A fixed synthetic batch produces identical memory embeddings, QA logits/loss rows, Recon stats, and gradients under the same seed.
- `filter_rows_batch` imported through both cache-builder and dataset paths returns identical rows and lengths.
- `TrainConfig.from_file(...).to_dict()` remains stable across the split modules.
- `Evaluator.compare_reader_tasks` reuses exactly one deterministic selection for QA and Recon.

Verification commands:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate shine
pytest -q
ruff check .
git diff --check
node .gitnexus/run.cjs analyze --index-only --pdg
node .gitnexus/run.cjs detect-changes --scope all --repo .
```

## 9. Risk and Impact Analysis

- **Model risk: HIGH operational impact despite MEDIUM graph risk.** LoRA enable/disable state, dtype casts, hooks, transformer-body caching, and `past_key_values` are mutable hidden state. Use delegation first and compare outputs/gradients before deleting original code.
- **Data risk: HIGH.** `filter_rows_batch` has HIGH GitNexus risk and feeds both cache building and runtime loading. Any duplicate implementation can silently change the dataset. Keep one implementation and test serial/parallel byte identity.
- **Training risk: HIGH.** `main` owns optimizer-step boundaries, DDP barriers, scheduler stepping, checkpoint ordering, and stage transitions. The loop must receive explicit dependencies and preserve collective ordering; no rank may skip a collective path.
- **Configuration risk: MEDIUM.** YAML inheritance and environment expansion feed almost every script. Preserve serialized field names and validation order; avoid changing default values during the move.
- **Evaluation risk: MEDIUM.** Sampling and distributed reduction affect reported research claims. Preserve selection seeds, row bases, dataset buckets, metric names, and all-gather/reduction behavior.
- **Migration risk.** New modules can create import cycles if they import facades. New leaf modules must depend downward; facades may import leaves, never the reverse.
- **Performance risk.** Moving code must not add tensor copies, extra tokenizer calls, extra transformer passes, or disable cache reuse. Keep hot-path helpers free of logging and filesystem work.
- **Observability.** Preserve training schedule output, metric keys, checkpoint metadata, and error messages used by tests and run analysis.

## 10. Files Expected to Change

| File | Symbols | Reason |
| ---- | ------- | ------ |
| `src/model.py` | `ContextMemoryModel`, state dataclasses, `load_model` | Facade and stable public API |
| `src/model_backbone.py` | LoRA, masks, positions, backbone helpers | Extract model infrastructure |
| `src/model_memory.py` | context encoding and prefix creation | Isolate memory writer |
| `src/model_readout.py` | readout assembly | Isolate question-conditioned reader |
| `src/model_tasks.py` | QA/Recon forward paths | Isolate supervision |
| `src/model_generation.py` | generation/cache helpers | Isolate decode path |
| `src/model_conversation.py` | conversation state/recompression | Isolate mutable session path |
| `src/data.py` | public re-exports/facade | Preserve imports |
| `src/data_records.py` | records and rendering | Isolate data model |
| `src/data_filter.py` | filtering/process pool | Preserve one canonical filter |
| `src/data_sampling.py` | lengths/sampler/turn sampling | Isolate deterministic sampling |
| `src/data_collate.py` | collate and packing | Isolate tensor batch construction |
| `src/data_dataset.py` | dataset/cache reads | Isolate persistence boundary |
| `src/cache_build.py` | imports/orchestration | Consume canonical filter module |
| `scripts/train.py` | CLI `main` | Thin composition root |
| `src/training_schedule.py` | pure schedule helpers | Reusable/testable policy functions |
| `src/training_setup.py` | setup and dependency construction | Isolate initialization |
| `src/training_resume.py` | resume/path state | Isolate checkpoint startup |
| `src/training_evaluation.py` | validation/checkpoint evaluation | Isolate evaluation integration |
| `src/training_loop.py` | loop and step semantics | Isolate mutable training state |
| `utils/config.py` | public config API | Thin facade |
| `utils/config_loader.py` | YAML/env/group loading | Isolate parsing |
| `utils/config_schema.py` | dataclasses | Isolate schema |
| `utils/config_validation.py` | validation rules | Isolate invariants |
| `src/evaluator.py` | `Evaluator` facade | Preserve evaluation API |
| `src/evaluation_distributed.py` | distributed utilities | Isolate collective helpers |
| `src/evaluation_qa.py` | QA metrics | Isolate QA evaluation |
| `src/evaluation_recon.py` | Recon metrics | Isolate Recon evaluation |
| `src/evaluation_sampling.py` | shared selection/orchestration | Preserve sample reuse |
| `tests/*` listed in §8 | existing and contract tests | Regression coverage |

## 11. Reusable Implementation Context

```yaml
implementation_context:
  task_summary: "Split oversized model, data, training, config, and evaluation files by responsibility while preserving all current behavior and public entrypoints."
  acceptance_criteria:
    - "No behavior change in model outputs, gradients, cache bytes, metrics, schedules, or checkpoint round-trips."
    - "Existing public imports and CLI commands remain valid."
    - "Core files become facades/composition roots and new modules have one-way dependency flow."
    - "pytest, ruff, git diff --check, refreshed GitNexus analysis, and detect_changes pass."
  evidence_provenance:
    schema_version: 2
    head_commit: '3979ac11ca347e0c410d2493509da0f5cb5fd2d4'
    generated_plan_path: 'docs/plans/2026-10-25-gitnexus-plan-split-core-files.md'
    global_dirty_digest:
      algorithm: 'sha256'
      canonicalization: 'gitnexus-evidence-provenance-v2 NUL-framed UTF-8 records'
      value: '0a9c85780067d9afcd0764f307b60891e3cee927ee11eaeb5ec7826d10fd82cd'
    cited_path_manifest:
      - path: '.github/workflows/ci.yml'
        object_kind: {"head": "regular", "index": "regular", "worktree": "regular", "untracked": "absent"}
        state: 'clean'
        rename_from: null
        rename_to: null
        head_digest: 'sha256:e4c103ef362bdf19a29be4fddc5f4e644deaf1ae70a262e77ce401dfd9fb81c9'
        index_digest: 'sha256:e4c103ef362bdf19a29be4fddc5f4e644deaf1ae70a262e77ce401dfd9fb81c9'
        worktree_digest: 'sha256:e4c103ef362bdf19a29be4fddc5f4e644deaf1ae70a262e77ce401dfd9fb81c9'
        untracked_digest: 'absent'
      - path: 'AGENTS.md'
        object_kind: {"head": "regular", "index": "regular", "worktree": "regular", "untracked": "absent"}
        state: 'clean'
        rename_from: null
        rename_to: null
        head_digest: 'sha256:ee9eaeeed773eee7d571dad87af5c32413f0f9b7d9601f4acaa6716cf41548fc'
        index_digest: 'sha256:ee9eaeeed773eee7d571dad87af5c32413f0f9b7d9601f4acaa6716cf41548fc'
        worktree_digest: 'sha256:ee9eaeeed773eee7d571dad87af5c32413f0f9b7d9601f4acaa6716cf41548fc'
        untracked_digest: 'absent'
      - path: 'pyproject.toml'
        object_kind: {"head": "regular", "index": "regular", "worktree": "regular", "untracked": "absent"}
        state: 'clean'
        rename_from: null
        rename_to: null
        head_digest: 'sha256:42dee2d3a439abe91806402dc95021623827333f496bb2ab12dfb9d948ae50d5'
        index_digest: 'sha256:42dee2d3a439abe91806402dc95021623827333f496bb2ab12dfb9d948ae50d5'
        worktree_digest: 'sha256:42dee2d3a439abe91806402dc95021623827333f496bb2ab12dfb9d948ae50d5'
        untracked_digest: 'absent'
      - path: 'scripts/train.py'
        object_kind: {"head": "regular", "index": "regular", "worktree": "regular", "untracked": "absent"}
        state: 'clean'
        rename_from: null
        rename_to: null
        head_digest: 'sha256:002ed2f4b7c539e54dfad5de4f3122b176940dea7aaf9d88fe016f88a96badf0'
        index_digest: 'sha256:002ed2f4b7c539e54dfad5de4f3122b176940dea7aaf9d88fe016f88a96badf0'
        worktree_digest: 'sha256:002ed2f4b7c539e54dfad5de4f3122b176940dea7aaf9d88fe016f88a96badf0'
        untracked_digest: 'absent'
      - path: 'src/cache_build.py'
        object_kind: {"head": "regular", "index": "regular", "worktree": "regular", "untracked": "absent"}
        state: 'clean'
        rename_from: null
        rename_to: null
        head_digest: 'sha256:e877fa0b3924c085cf6dcbd4148aa903d9b8ea3488c15ed17a8ecc2cdf55f559'
        index_digest: 'sha256:e877fa0b3924c085cf6dcbd4148aa903d9b8ea3488c15ed17a8ecc2cdf55f559'
        worktree_digest: 'sha256:e877fa0b3924c085cf6dcbd4148aa903d9b8ea3488c15ed17a8ecc2cdf55f559'
        untracked_digest: 'absent'
      - path: 'src/data.py'
        object_kind: {"head": "regular", "index": "regular", "worktree": "regular", "untracked": "absent"}
        state: 'clean'
        rename_from: null
        rename_to: null
        head_digest: 'sha256:dbd919aaf72990b74dfc9c470868c69d7ffc1bc6ef756be92800764e0a224dc7'
        index_digest: 'sha256:dbd919aaf72990b74dfc9c470868c69d7ffc1bc6ef756be92800764e0a224dc7'
        worktree_digest: 'sha256:dbd919aaf72990b74dfc9c470868c69d7ffc1bc6ef756be92800764e0a224dc7'
        untracked_digest: 'absent'
      - path: 'src/evaluator.py'
        object_kind: {"head": "regular", "index": "regular", "worktree": "regular", "untracked": "absent"}
        state: 'clean'
        rename_from: null
        rename_to: null
        head_digest: 'sha256:a912c7573dea666d1e65b11e0639556aeb2e39944db31d1b65f8f9bf16b7b539'
        index_digest: 'sha256:a912c7573dea666d1e65b11e0639556aeb2e39944db31d1b65f8f9bf16b7b539'
        worktree_digest: 'sha256:a912c7573dea666d1e65b11e0639556aeb2e39944db31d1b65f8f9bf16b7b539'
        untracked_digest: 'absent'
      - path: 'src/losses.py'
        object_kind: {"head": "regular", "index": "regular", "worktree": "regular", "untracked": "absent"}
        state: 'clean'
        rename_from: null
        rename_to: null
        head_digest: 'sha256:0c51aa5f5359797d1f6ca200972bc53ba02e5b2bd1c3c287b9c8b3bd80ffcecd'
        index_digest: 'sha256:0c51aa5f5359797d1f6ca200972bc53ba02e5b2bd1c3c287b9c8b3bd80ffcecd'
        worktree_digest: 'sha256:0c51aa5f5359797d1f6ca200972bc53ba02e5b2bd1c3c287b9c8b3bd80ffcecd'
        untracked_digest: 'absent'
      - path: 'src/model.py'
        object_kind: {"head": "regular", "index": "regular", "worktree": "regular", "untracked": "absent"}
        state: 'clean'
        rename_from: null
        rename_to: null
        head_digest: 'sha256:9df2b8a7f6aed4e3a67b275defd8fec44036f78c1297b5419d5179f059726417'
        index_digest: 'sha256:9df2b8a7f6aed4e3a67b275defd8fec44036f78c1297b5419d5179f059726417'
        worktree_digest: 'sha256:9df2b8a7f6aed4e3a67b275defd8fec44036f78c1297b5419d5179f059726417'
        untracked_digest: 'absent'
      - path: 'src/pipeline.py'
        object_kind: {"head": "regular", "index": "regular", "worktree": "regular", "untracked": "absent"}
        state: 'clean'
        rename_from: null
        rename_to: null
        head_digest: 'sha256:bcd7e6a2729e112fd6a85e7476836b8d55af9c1f0d65fe74ff522a1b4b820b17'
        index_digest: 'sha256:bcd7e6a2729e112fd6a85e7476836b8d55af9c1f0d65fe74ff522a1b4b820b17'
        worktree_digest: 'sha256:bcd7e6a2729e112fd6a85e7476836b8d55af9c1f0d65fe74ff522a1b4b820b17'
        untracked_digest: 'absent'
      - path: 'src/resampler.py'
        object_kind: {"head": "regular", "index": "regular", "worktree": "regular", "untracked": "absent"}
        state: 'clean'
        rename_from: null
        rename_to: null
        head_digest: 'sha256:79f7afb826c1df6e1c5acdf388785d42bbddc5aff361b54b6e1dc9e90df89884'
        index_digest: 'sha256:79f7afb826c1df6e1c5acdf388785d42bbddc5aff361b54b6e1dc9e90df89884'
        worktree_digest: 'sha256:79f7afb826c1df6e1c5acdf388785d42bbddc5aff361b54b6e1dc9e90df89884'
        untracked_digest: 'absent'
      - path: 'tests/test_build_cache.py'
        object_kind: {"head": "regular", "index": "regular", "worktree": "regular", "untracked": "absent"}
        state: 'clean'
        rename_from: null
        rename_to: null
        head_digest: 'sha256:391adba2c42e7cc999958c2613cd79ba10b7f55c56f7facffb4b773e598a9c4d'
        index_digest: 'sha256:391adba2c42e7cc999958c2613cd79ba10b7f55c56f7facffb4b773e598a9c4d'
        worktree_digest: 'sha256:391adba2c42e7cc999958c2613cd79ba10b7f55c56f7facffb4b773e598a9c4d'
        untracked_digest: 'absent'
      - path: 'tests/test_chat_template_boundaries.py'
        object_kind: {"head": "regular", "index": "regular", "worktree": "regular", "untracked": "absent"}
        state: 'clean'
        rename_from: null
        rename_to: null
        head_digest: 'sha256:2de79806eec6f2e04c36eaf14ee950a79e4b2ca8c26a9b1fe9bb51670cde24b7'
        index_digest: 'sha256:2de79806eec6f2e04c36eaf14ee950a79e4b2ca8c26a9b1fe9bb51670cde24b7'
        worktree_digest: 'sha256:2de79806eec6f2e04c36eaf14ee950a79e4b2ca8c26a9b1fe9bb51670cde24b7'
        untracked_digest: 'absent'
      - path: 'tests/test_context_writer.py'
        object_kind: {"head": "regular", "index": "regular", "worktree": "regular", "untracked": "absent"}
        state: 'clean'
        rename_from: null
        rename_to: null
        head_digest: 'sha256:a6a8db45077d5adc12b3e66bded598b34be1cc7dccd73b8308dc82c72249ee15'
        index_digest: 'sha256:a6a8db45077d5adc12b3e66bded598b34be1cc7dccd73b8308dc82c72249ee15'
        worktree_digest: 'sha256:a6a8db45077d5adc12b3e66bded598b34be1cc7dccd73b8308dc82c72249ee15'
        untracked_digest: 'absent'
      - path: 'tests/test_entrypoints.py'
        object_kind: {"head": "regular", "index": "regular", "worktree": "regular", "untracked": "absent"}
        state: 'clean'
        rename_from: null
        rename_to: null
        head_digest: 'sha256:6d7446fdd78fcbc2fed885876c0d531c5fbd0792d5fb34379ce00dc150a4a89d'
        index_digest: 'sha256:6d7446fdd78fcbc2fed885876c0d531c5fbd0792d5fb34379ce00dc150a4a89d'
        worktree_digest: 'sha256:6d7446fdd78fcbc2fed885876c0d531c5fbd0792d5fb34379ce00dc150a4a89d'
        untracked_digest: 'absent'
      - path: 'tests/test_generation.py'
        object_kind: {"head": "regular", "index": "regular", "worktree": "regular", "untracked": "absent"}
        state: 'clean'
        rename_from: null
        rename_to: null
        head_digest: 'sha256:1a920ffbc5f45e9c581f19ba380b3749c60f63899f7e6035c564e9dee71ad165'
        index_digest: 'sha256:1a920ffbc5f45e9c581f19ba380b3749c60f63899f7e6035c564e9dee71ad165'
        worktree_digest: 'sha256:1a920ffbc5f45e9c581f19ba380b3749c60f63899f7e6035c564e9dee71ad165'
        untracked_digest: 'absent'
      - path: 'tests/test_positions.py'
        object_kind: {"head": "regular", "index": "regular", "worktree": "regular", "untracked": "absent"}
        state: 'clean'
        rename_from: null
        rename_to: null
        head_digest: 'sha256:3f07f78114682a00bacdb9502ca8a2c5d2f7f0f03d9bf0bfa3eebd46b78a6101'
        index_digest: 'sha256:3f07f78114682a00bacdb9502ca8a2c5d2f7f0f03d9bf0bfa3eebd46b78a6101'
        worktree_digest: 'sha256:3f07f78114682a00bacdb9502ca8a2c5d2f7f0f03d9bf0bfa3eebd46b78a6101'
        untracked_digest: 'absent'
      - path: 'tests/test_resume.py'
        object_kind: {"head": "regular", "index": "regular", "worktree": "regular", "untracked": "absent"}
        state: 'clean'
        rename_from: null
        rename_to: null
        head_digest: 'sha256:fc522f84ace39946ba2666739b7ea52b0b470ed0e612d1f635fadc2d11cdfb1d'
        index_digest: 'sha256:fc522f84ace39946ba2666739b7ea52b0b470ed0e612d1f635fadc2d11cdfb1d'
        worktree_digest: 'sha256:fc522f84ace39946ba2666739b7ea52b0b470ed0e612d1f635fadc2d11cdfb1d'
        untracked_digest: 'absent'
      - path: 'tests/test_train_schedule.py'
        object_kind: {"head": "regular", "index": "regular", "worktree": "regular", "untracked": "absent"}
        state: 'clean'
        rename_from: null
        rename_to: null
        head_digest: 'sha256:970de08b5301682035d8429cdefcc8a2201251780d2ee3998e316bfde87a0eca'
        index_digest: 'sha256:970de08b5301682035d8429cdefcc8a2201251780d2ee3998e316bfde87a0eca'
        worktree_digest: 'sha256:970de08b5301682035d8429cdefcc8a2201251780d2ee3998e316bfde87a0eca'
        untracked_digest: 'absent'
      - path: 'utils/checkpoint.py'
        object_kind: {"head": "regular", "index": "regular", "worktree": "regular", "untracked": "absent"}
        state: 'clean'
        rename_from: null
        rename_to: null
        head_digest: 'sha256:4fcd51cecd1a7bce35de2b0d1f0066bf6702766f24473bed23b9bf33affd4ace'
        index_digest: 'sha256:4fcd51cecd1a7bce35de2b0d1f0066bf6702766f24473bed23b9bf33affd4ace'
        worktree_digest: 'sha256:4fcd51cecd1a7bce35de2b0d1f0066bf6702766f24473bed23b9bf33affd4ace'
        untracked_digest: 'absent'
      - path: 'utils/config.py'
        object_kind: {"head": "regular", "index": "regular", "worktree": "regular", "untracked": "absent"}
        state: 'clean'
        rename_from: null
        rename_to: null
        head_digest: 'sha256:94e83b7f7ec3fb5d36cf7b09d63812b2d223a5046ce6619a3c681f01d5f629b5'
        index_digest: 'sha256:94e83b7f7ec3fb5d36cf7b09d63812b2d223a5046ce6619a3c681f01d5f629b5'
        worktree_digest: 'sha256:94e83b7f7ec3fb5d36cf7b09d63812b2d223a5046ce6619a3c681f01d5f629b5'
        untracked_digest: 'absent'
  primary_symbols:
    - {symbol: "ContextMemoryModel", file: "src/model.py", lines: "134-1335", role: "model facade and state boundary"}
    - {symbol: "filter_rows_batch", file: "src/data.py", lines: "37-95", role: "canonical filter implementation"}
    - {symbol: "main", file: "scripts/train.py", lines: "306-1063", role: "training composition root"}
    - {symbol: "TrainConfig", file: "utils/config.py", lines: "387-637", role: "configuration facade"}
    - {symbol: "Evaluator", file: "src/evaluator.py", lines: "114-605", role: "evaluation facade"}
  related_symbols:
    - {symbol: "ContextPrefix", relationship: "state-carrier", relevance: "shared model handoff"}
    - {symbol: "ConversationState", relationship: "state-carrier", relevance: "mutable conversation cache"}
    - {symbol: "ensure_cache", relationship: "CALLS filter_rows_batch", relevance: "cache/runtime consistency"}
    - {symbol: "ConversationDataset", relationship: "CALLS filter_rows_batch", relevance: "runtime filtering"}
    - {symbol: "CheckpointManager", relationship: "used-by training", relevance: "resume/save boundary"}
    - {symbol: "resampler.py", relationship: "IMPORTS model", relevance: "existing cohesive leaf module"}
  execution_path:
    - "CLI loads TrainConfig and resolves machine/environment paths."
    - "Training setup loads tokenizer/model, ensures filtered cache, builds datasets/loaders, optimizer, scheduler, and checkpoint managers."
    - "Training loop samples task masks, computes QA/Recon objectives, steps optimizer and scheduler, evaluates, logs, and saves checkpoints."
    - "Model context encoding produces ContextPrefix; task/generation/conversation paths consume it with LoRA and cache state rules."
    - "Evaluation reuses deterministic sampled contexts for QA and Recon and reduces metrics across ranks."
  pdg_constraints:
    - {description: "Task fallback sampling is a single guarded operation.", affected_statements: ["scripts/train.py:102-116"], implementation_consequence: "Keep random generator and fallback probability together."}
    - {description: "Optimizer, scheduler, logging, and progress updates share the optimizer-step boundary.", affected_statements: ["scripts/train.py:817-883"], implementation_consequence: "Do not move scheduler into Accelerate preparation or split the step transaction."}
    - {description: "Filtering selects bounded process-pool or serial execution.", affected_statements: ["src/data.py:224-251"], implementation_consequence: "Retain one canonical filter and bounded pending futures."}
    - {description: "Config values feed all section constructors after one expansion pass.", affected_statements: ["utils/config.py:432-488"], implementation_consequence: "Preserve normalization, error handling, and construction order."}
  architectural_patterns:
    - {pattern: "Facade over leaf modules", example_location: "src/pipeline.py", usage_guidance: "Keep existing import paths while moving implementation downward."}
    - {pattern: "Single source of truth for filtering", example_location: "src/data.py:37-95", usage_guidance: "Cache builder and loader must share the same function."}
    - {pattern: "Explicit state carriers", example_location: "src/model.py:27-44", usage_guidance: "Pass ContextPrefix/ConversationState rather than hidden globals."}
  files_to_modify:
    - {file: "src/model.py and src/model_*.py", symbols: ["ContextMemoryModel", "ContextPrefix", "ConversationState"], intended_change: "Split model concerns and preserve facade methods."}
    - {file: "src/data.py and src/data_*.py", symbols: ["filter_rows_batch", "ConversationDataset", "collate_context_records"], intended_change: "Split records, filtering, sampling, dataset, and collation."}
    - {file: "scripts/train.py and src/training_*.py", symbols: ["main", "training_stage", "sample_task_masks"], intended_change: "Split composition, setup, resume, evaluation, and loop."}
    - {file: "utils/config.py and utils/config_*.py", symbols: ["TrainConfig"], intended_change: "Split loader, schema, and validation behind stable API."}
    - {file: "src/evaluator.py and src/evaluation_*.py", symbols: ["Evaluator"], intended_change: "Split distributed, QA, Recon, and sampling concerns."}
  tests:
    - {file: "tests/test_context_writer.py", scenarios: ["same memory outputs/gradients after model delegation", "LoRA state restored after failure"]}
    - {file: "tests/test_generation.py", scenarios: ["same greedy tokens", "same cache isolation", "same conversation recompression"]}
    - {file: "tests/test_build_cache.py", scenarios: ["serial/parallel byte identity", "same cache fingerprint and sidecar"]}
    - {file: "tests/test_resume.py", scenarios: ["same two-stage boundary", "same skipped micro-batches"]}
    - {file: "tests/test_entrypoints.py", scenarios: ["same config/checkpoint round-trip", "same machine relocation"]}
  verification_commands:
    - "source $(conda info --base)/etc/profile.d/conda.sh && conda activate shine && pytest -q"
    - "source $(conda info --base)/etc/profile.d/conda.sh && conda activate shine && ruff check ."
    - "git diff --check"
    - "node .gitnexus/run.cjs analyze --index-only --pdg"
    - "node .gitnexus/run.cjs detect-changes --scope all --repo ."
  risks:
    - "LoRA/dtype/hooks/cache state can change silently during extraction."
    - "filter_rows_batch has HIGH impact and must remain canonical."
    - "DDP collective ordering and scheduler cadence are sensitive to loop extraction."
    - "Config and evaluation changes can invalidate experiment comparability."
  assumptions:
    - "Current public imports listed by GitNexus remain required until migration is complete; re-check before deleting facade exports."
    - "The current CPU test suite is the baseline oracle; add GPU smoke comparison only if model delegation changes numerical behavior."
  open_questions:
    - "Choose flat modules versus new subpackages before implementation step 2; prefer flat modules for the smallest import migration."
    - "Decide whether pure training helpers remain under scripts/train.py for test compatibility or move to src/training_schedule.py with explicit re-exports."
  avoid:
    - "Do not change algorithms, tensor layouts, cache formats, config keys, metric names, or CLI flags during the structural refactor."
    - "Do not duplicate filter, prompt, schedule, or validation logic."
    - "Do not introduce reverse imports from leaf modules into facades."
    - "Do not add broad legacy compatibility code; preserve only current public import paths required by existing callers."
    - "Do not edit production code before running impact for the target symbol."
    - "Do not commit before detect_changes reports complete, non-truncated results."

## 12. Assumptions and Open Questions

Assumptions:

- [assumed] Existing public imports from `src.model`, `src.data`, `utils.config`, and `src.evaluator` remain in use until each migration stage is complete; verify with fresh GitNexus/text search before removing any facade export.
- [assumed] Flat new modules are preferred for the first pass because they minimize package/import churn; a package hierarchy can be introduced later if the flat layout remains crowded.
- [assumed] Numerical equivalence is judged with existing CPU fixtures and deterministic seeds; GPU smoke validation is added if dtype or cache comparisons become sensitive.

Open questions:

- Choose flat modules versus new subpackages before implementation step 2.
- Decide whether to preserve private helper names through facade imports or update tests to import the new owner modules; do not keep duplicate implementations.
- Decide whether `scripts/train.py` should remain the only CLI file or gain a `python -m` training package entrypoint; this plan keeps the current CLI.

Explicitly deferred: experiment-shell consolidation, external path cleanup, and changes to the training algorithm or configuration schema.

## 13. Definition of Done

- All five original core files are reduced to facades/composition roots with no duplicated implementation blocks.
- New modules have one-way imports and each has one documented responsibility.
- Existing imports, CLI invocations, checkpoint formats, cache paths, metric keys, and config keys remain valid.
- Focused model/data/training/config/evaluation tests pass after each extraction stage.
- Full `pytest -q`, `ruff check .`, and `git diff --check` pass in `shine`.
- GitNexus is refreshed at the final tree; `detect_changes --scope all` is complete and non-truncated, and all HIGH/CRITICAL impacts have been reviewed.
- The final diff contains only the structural refactor, its contract tests, and necessary facade/import updates.
