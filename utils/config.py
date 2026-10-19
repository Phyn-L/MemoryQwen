from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .machines import fill_missing, machine_environ, resolve_machine

# ``${VAR}`` or ``${VAR:-default}`` inside a configuration string.
ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge nested configuration mappings; scalar and list values are replaced."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_with_base(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load YAML and recursively merge the optional relative ``_base_`` config."""
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"configuration inheritance cycle: {chain}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"configuration must be a YAML mapping: {path}")
    base_spec = raw.pop("_base_", None)
    if base_spec is None:
        return dict(raw)
    base_paths = [base_spec] if isinstance(base_spec, str) else base_spec
    if not isinstance(base_paths, list) or not all(
        isinstance(item, str) for item in base_paths
    ):
        raise ValueError(f"_base_ must be a path or list of paths: {path}")
    merged: dict[str, Any] = {}
    for base in base_paths:
        base_path = (path.parent / base).resolve()
        merged = _deep_merge(merged, _load_yaml_with_base(base_path, (*stack, path)))
    return _deep_merge(merged, raw)


def _flatten_data_groups(data: Mapping[str, Any]) -> dict[str, Any]:
    """Accept grouped YAML data settings while keeping the flat DataConfig API.

    A key that is a ``DataConfig`` field keeps its value (even a mapping); any other mapping is
    a group (``files``/``limits``/``loader``/``task``) and is unfolded. Field-aware unfolding is
    what makes unknown fields fail rather than silently scattering into the data namespace.
    """
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Mapping) and key not in DataConfig.__dataclass_fields__:
            nested = _flatten_data_groups(value)
            duplicate = set(flattened).intersection(nested)
            if duplicate:
                raise ValueError(
                    f"duplicate data fields after grouping: {sorted(duplicate)}"
                )
            flattened.update(nested)
        else:
            if key in flattened:
                raise ValueError(f"duplicate data field: {key}")
            flattened[key] = value
    if not flattened:
        return flattened
    unknown = sorted(set(flattened) - set(DataConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"unknown data field(s): {', '.join(unknown)}")
    return flattened


def expand_env(value: str, overrides: Mapping[str, str] | None = None) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}`` references in one configuration string.

    Machine-specific locations must not be baked into tracked files. The model directory and
    the data root live somewhere different on every cluster, and editing the YAML
    there turns every ``git pull`` into a conflict -- which is exactly what happened. Writing
    them as ``${MODEL_ROOT:-/data/lz/hf_cache/hub}`` keeps the default valid where it was
    written and lets another machine override it, either through the environment (a gitignored
    ``scripts/env.local.sh``, a one-off ``export``) or through the per-machine table in
    ``utils/machines.py``, which a single ``machine`` field selects.

    A bare ``${VAR}`` with no fallback raises when the variable is unset, so a typo cannot
    silently turn into an empty path.

    ``overrides`` wins over the process environment: a machine named on the command line or in
    the config decides where the weights and the data are for this run, instead of depending on
    whatever the shell happens to export.
    """
    overrides = overrides or {}

    def replace(match: re.Match) -> str:
        name, default = match.group(1), match.group(2)
        if overrides.get(name):
            return overrides[name]
        if os.environ.get(name):
            return os.environ[name]
        if default is not None:
            return default
        raise ValueError(
            f"${{{name}}} is not set and has no default; export {name} or write "
            f"${{{name}:-<default>}} in the configuration"
        )

    return ENV_REFERENCE.sub(replace, value)


def expand_env_values(value: Any, overrides: Mapping[str, str] | None = None) -> Any:
    """Apply :func:`expand_env` to every string in a nested configuration structure."""
    if isinstance(value, str):
        return expand_env(value, overrides)
    if isinstance(value, dict):
        return {key: expand_env_values(item, overrides) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [expand_env_values(item, overrides) for item in value]
    return value


@dataclass
class ModelConfig:
    name_or_path: str = (
        "/data/lz/hf_cache/hub/models--Qwen--Qwen3-1.7B/"
        "snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    )
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    context_gradient_checkpointing: bool = True
    # Dtype of the trainable pieces (memory writer, LoRA, recon decoders).
    # Kept separate from torch_dtype so that AdamW keeps float32 parameters and
    # moments while the frozen backbone stays at its checkpoint dtype.
    trainable_dtype: str = "float32"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass
class MemoryConfig:
    memory_length: int = 8
    # Per-question read-out over the cached (question-agnostic) memory: R learnable latents
    # cross-attend over the question and the memory and are inserted as R extra input
    # positions between the question and the answer. 0 = off (the historical behaviour).
    readout_mode: str = "memory_question"
    readout_length: int = 0
    readout_layers: int = 2
    readout_heads: int = 4
    # Bottleneck width of the question-conditioned resampler.
    readout_hidden_size: int = 256
    # Context encoder -> fixed memory writer.
    context_layer_indices: tuple[int, ...] = (-1,)
    memory_resampler_layers: int = 2
    memory_resampler_heads: int = 4
    memory_resampler_hidden_size: int = 256
    memory_resampler_ffn_ratio: int = 2
    memory_resampler_output_mode: str = "layer0"
    # Source exposed to the optional question-conditioned reader.
    reader_memory_source: str = "memory_hidden"


@dataclass
class LossConfig:
    """Every knob the objective reads.

    Kept apart from :class:`MemoryConfig` so that the *architecture* (slots, resampler
    width, read-out) and the *objective* (weights, distillation budget, logits chunk size)
    can be read, swept and diffed independently.
    """

    qa_weight: float = 1.0
    recon_weight: float = 1.0
    # Distil the *full-context* distribution into the memory-conditioned one:
    # T^2 * KL(p_teacher || p_student) where the teacher is the plain causal LM over the
    # context (free: the encoder pass's context rows) and the student is the memory-prefixed
    # autoencoding pass. 0 = off. Runs the prefix-KV path independently of its CE weight.
    distill_weight: float = 0.0
    distill_temperature: float = 1.0
    # Positions scored by the distillation objective; <= 0 scores every position.
    distill_positions: int = 256
    # Weight each position by the teacher's own entropy instead of uniformly.
    distill_entropy_weight: bool = False
    # Truncate the teacher distribution to its k most likely tokens (0 = keep all).
    distill_topk: int = 0
    # Vocabulary rows projected per chunk. Bounds the [rows, vocab] logits that stay alive
    # for the backward pass; the whole batch is still projected in one flattened pass, so
    # this only trades peak memory against kernel count.
    max_logits_rows: int = 256


@dataclass
class DataConfig:
    """Where the JSONL data lives and how it is filtered.

    ``root`` is the data tree produced by ``scripts/data/``; the 1QA splits are read from
    ``<root>/1qa/1qa-{train,validation,test}.jsonl`` unless a split overrides its path.
    """

    root: str = "contexts"
    # Explicit per-split sources; None means "use <root>/1qa/1qa-<split>.jsonl".
    train_conversation_file: str | None = None
    validation_conversation_file: str | None = None
    test_conversation_file: str | None = None
    dataset: str = "all"
    train_datasets: tuple[str, ...] | None = None
    validation_datasets: tuple[str, ...] | None = None
    test_datasets: tuple[str, ...] | None = None
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    max_context_tokens: int = 2048
    # Reject source contexts before truncation, independently of the encoder limit.
    max_input_tokens: int = 40960
    train_tasks: str = "mix"
    # Recon loss fraction for constant mixed supervision.
    recon_ratio: float = 0.5
    recon_schedule: str = "constant"
    recon_schedule_curve: str = "linear"
    recon_ratio_start: float = 0.5
    recon_ratio_end: float = 0.5
    recon_ratio_min: float = 0.1
    max_question_tokens: int = 96
    max_answer_tokens: int = 64
    max_conversation_tokens: int = 768
    max_samples: int | None = None
    train_max_samples: int | None = None
    #: Deterministic cap on the validation contexts loaded (``None`` = all of them).
    validation_max_samples: int | None = None
    #: Where the loader's own preprocessing caches go (filtered rows for a file that is not the
    #: built cache, and the per-context lengths the sortish sampler buckets by). ``None`` puts
    #: them under ``<root>/1qa/cache/loader``; set it to local scratch when the data root is a
    #: slow network filesystem.
    preprocess_cache_dir: str | None = None
    filter_long_context: bool = True
    filter_no_qa: bool = True
    tokenizer_filter_batch_size: int = 2048
    sortish_bucket_multiplier: int = 50
    cache_sortish_lengths: bool = True
    cache_filtered_dataset: bool = True
    append_eos: bool = True
    chat_template_enable_thinking: bool = False
    max_qa_sample_per_context: int = 8
    # Left-padding keeps the question's real tokens adjacent to the answer; right
    # padding inserts pad positions between them, so the first answer token would be
    # supervised from a padding hidden state.  Set to "right" only to reproduce old runs.
    question_padding_side: str = "left"
    train_num_workers: int = 4
    train_pin_memory: bool = True
    train_prefetch_factor: int = 2
    train_persistent_workers: bool = True
    eval_num_workers: int = 2
    eval_pin_memory: bool = True
    eval_prefetch_factor: int = 2
    eval_persistent_workers: bool = True
    # "append" writes EOS after the last answer token; the legacy "overwrite" mode
    # overwrites that token whenever the answer is the longest in the batch.

    SPLIT_DIRNAME = "1qa"

    def meta_file(self, split: str) -> Path:
        """The raw, unfiltered split ``scripts/data/build_1qa.py`` writes.

        One line per context with every QA pair of that context: no length cap and no
        per-dataset row cap.
        """
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"unknown data split: {split!r}")
        return Path(self.root) / self.SPLIT_DIRNAME / "meta" / f"meta_{split}.json"

    def cache_file(self, split: str) -> Path:
        """The filtered split ``scripts/data/build_cache.py`` writes from the meta file."""
        if split not in {"train", "validation"}:
            raise ValueError(
                f"only train and validation have a filtered cache (asked {split!r})"
            )
        return Path(self.root) / self.SPLIT_DIRNAME / "cache" / f"{split}.jsonl"

    def split_file(self, split: str) -> str:
        """The file a training or evaluation run reads for one split.

        ``train`` and ``validation`` resolve to the filtered cache (built from the meta file
        with this config's filters); ``test`` resolves to the raw meta file because there is no
        filtered test product. An explicit ``<split>_conversation_file`` always wins.
        """
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"unknown data split: {split!r}")
        explicit = {
            "train": self.train_conversation_file,
            "validation": self.validation_conversation_file,
            "test": self.test_conversation_file,
        }[split]
        if explicit:
            return str(explicit)
        if split in {"train", "validation"}:
            return str(self.cache_file(split))
        return str(self.meta_file(split))

    def validation_source(self) -> str:
        """Validation file a run evaluates on: the filtered cache."""
        return self.split_file("validation")


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_steps: int = 500


@dataclass
class EvaluationConfig:
    autoregressive_every: int = 4000
    max_new_tokens: int = 128
    qa_batch_size: int = 4
    # Teacher-forced reconstruction score on the unique contexts referenced by the
    # selected QA sample. One memory-prefixed forward per context, no decoding.
    recon_teacherforce: bool = True
    # Autoregressive context reconstruction (``max_context_tokens`` decodes per context).
    # Off by default; ``recon_teacherforce`` is the cheap stand-in.
    recon_autoregressive: bool = False
    # Seeded context sample per dataset, across all ranks. No combined sample cap.
    # Each selected context contributes at most data.max_qa_sample_per_context QA turns.
    max_eval_contexts_per_dataset: int = 1280


@dataclass
class TrainingConfig:
    batch_size: int = 2
    # Micro-batches per optimizer step: the effective batch is
    # ``batch_size * gradient_accumulation_steps * world_size``. 1 keeps the historical
    # behaviour (one optimizer step per DataLoader batch).
    gradient_accumulation_steps: int = 1
    gradient_checkpoint_every_n_layers: int = 1
    epochs: int = 1
    max_grad_norm: float = 1.0
    seed: int = 42
    deterministic: bool = False


@dataclass
class CheckpointConfig:
    output_dir: str = "outputs"
    init_from: str | None = None


@dataclass
class LoggingConfig:
    swanlab_project: str = "MemoryQwen"
    # Online runs require an explicit private host (or SWANLAB_HOST).
    swanlab_host: str | None = None
    swanlab_mode: str = "online"
    swanlab_log_dir: str | None = (
        None  # Automatically set to the checkpoint run directory.
    )
    swanlab_workspace: str | None = None
    # Dashboard name; empty means model name + timestamp.
    swanlab_run_name: str | None = None
    swanlab_run_id: str | None = None
    # Cadence of the training-loss scalars (``train/*``), in optimizer steps. The two
    # evaluation sections keep their own cadence, so this only controls how densely the
    # loss curves are sampled. 10 is the historical value; a single-epoch run can afford
    # to be denser (or sparser) without touching any other schedule.
    log_every: int = 10


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    # Which entry of ``utils/machines.py::MACHINES`` supplied the paths (None = the config's
    # own defaults). Recorded so a run's SwanLab config and checkpoint say where it ran.
    machine: str | None = None

    @classmethod
    def from_file(cls, path: str | Path, machine: str | None = None) -> TrainConfig:
        """Load a config, resolving machine-specific paths.

        ``machine`` (or ``machine:`` in the file, or ``MACHINE`` in the environment) selects
        an entry from ``utils/machines.py::MACHINES``; a machine that was named outright
        supplies ``MODEL_ROOT``/``DATA_ROOT`` for this load, while one only
        recognised from the hostname just fills in what the environment did not set.
        """
        path = Path(path)
        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(
                f"configuration must be a YAML file (.yaml/.yml), got: {path}"
            )
        raw = _load_yaml_with_base(path)
        return cls.from_dict(raw, machine=machine)

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any],
        machine: str | None = None,
    ) -> TrainConfig:
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        resolved, authoritative = resolve_machine(machine or raw.get("machine"))
        overrides = machine_environ(resolved)
        if not authoritative:
            overrides = fill_missing(overrides, os.environ)
        values = expand_env_values(raw, overrides)
        values = dict(values)
        data_values = _flatten_data_groups(values.get("data", {}))
        for key in ("train_datasets", "validation_datasets", "test_datasets"):
            if isinstance(data_values.get(key), list):
                data_values[key] = tuple(data_values[key])
        optimizer_values = dict(values.get("optimizer", {}))
        if isinstance(optimizer_values.get("betas"), list):
            optimizer_values["betas"] = tuple(optimizer_values["betas"])
        model_values = dict(values.get("model", {}))
        if isinstance(model_values.get("target_modules"), list):
            model_values["target_modules"] = tuple(model_values["target_modules"])
        memory_values = dict(values.get("memory", {}))
        # ``off`` is a YAML 1.1 boolean, so a bare `readout_mode: off` arrives as False and
        # used to fail validation with "invalid memory.readout_mode". The documented value is
        # the string name; accept the boolean spelling too.
        if memory_values.get("readout_mode") is False:
            memory_values["readout_mode"] = "off"
        if isinstance(memory_values.get("context_layer_indices"), list):
            memory_values["context_layer_indices"] = tuple(
                memory_values["context_layer_indices"]
            )
        # The objective moved out of ``memory:``; a stale YAML must say so instead of
        # silently training with the default weights.
        moved = sorted(set(memory_values).intersection(LossConfig.__dataclass_fields__))
        if moved:
            raise ValueError(
                f"memory.{', memory.'.join(moved)} moved to the loss section; "
                f"move {'these fields' if len(moved) > 1 else 'this field'} under `loss:`"
            )
        checkpoint_values = dict(values.get("checkpoint", {}))
        checkpoint_values.pop("save_every_steps", None)
        logging_values = dict(values.get("logging", {}))
        evaluation_values = dict(values.get("evaluation", {}))
        if "recon_max_contexts" in evaluation_values:
            raise ValueError(
                "evaluation.recon_max_contexts was removed; Recon now uses contexts "
                "from the sampled QA rows"
            )
        return cls(
            model=ModelConfig(**model_values),
            memory=MemoryConfig(**memory_values),
            loss=LossConfig(**values.get("loss", {})),
            data=DataConfig(**data_values),
            optimizer=OptimizerConfig(**optimizer_values),
            scheduler=SchedulerConfig(**values.get("scheduler", {})),
            evaluation=EvaluationConfig(**evaluation_values),
            training=TrainingConfig(**values.get("training", {})),
            checkpoint=CheckpointConfig(**checkpoint_values),
            logging=LoggingConfig(**logging_values),
            machine=resolved,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        d, m, loss = self.data, self.memory, self.loss
        if self.model.attn_implementation not in {"sdpa", "eager"}:
            raise ValueError("model.attn_implementation must be sdpa or eager")
        if d.max_context_tokens <= 0:
            raise ValueError("data.max_context_tokens must be positive")
        if not str(d.root).strip():
            raise ValueError("data.root must point at the scripts/data output tree")
        if d.max_input_tokens <= 0:
            raise ValueError("data.max_input_tokens must be positive")
        if d.max_input_tokens < d.max_context_tokens:
            raise ValueError("data.max_input_tokens must be >= data.max_context_tokens")
        if d.tokenizer_filter_batch_size <= 0:
            raise ValueError("data.tokenizer_filter_batch_size must be positive")
        for name in ("train_num_workers", "eval_num_workers"):
            if getattr(d, name) < 0:
                raise ValueError(f"data.{name} must be non-negative")
        for name in ("train_prefetch_factor", "eval_prefetch_factor"):
            if getattr(d, name) <= 0:
                raise ValueError(f"data.{name} must be positive")
        if d.train_tasks not in {"mix", "qa", "recon"}:
            raise ValueError("data.train_tasks must be mix, qa or recon")
        if not 0 <= d.recon_ratio <= 1:
            raise ValueError("data.recon_ratio must be within [0, 1]")
        if d.recon_schedule not in {"decay", "front_loaded", "constant"}:
            raise ValueError(
                "data.recon_schedule must be decay, front_loaded or constant"
            )
        if d.recon_schedule_curve not in {"linear", "cosine"}:
            raise ValueError("data.recon_schedule_curve must be linear or cosine")
        for name in ("recon_ratio_start", "recon_ratio_end", "recon_ratio_min"):
            value = getattr(d, name)
            if not 0 <= value <= 1:
                raise ValueError(f"data.{name} must be within [0, 1]")
        if d.train_tasks == "mix" and d.recon_ratio_min <= 0:
            raise ValueError("mixed training requires data.recon_ratio_min > 0")
        if d.train_tasks == "mix":
            if d.recon_schedule == "constant":
                if not d.recon_ratio_min <= d.recon_ratio < 1:
                    raise ValueError(
                        "constant mixed ratio must be >= recon_ratio_min and < 1"
                    )
            else:
                if (
                    d.recon_ratio_start < d.recon_ratio_min
                    or d.recon_ratio_end < d.recon_ratio_min
                ):
                    raise ValueError(
                        "recon schedule endpoints must be >= recon_ratio_min"
                    )
                if d.recon_ratio_end > d.recon_ratio_start:
                    raise ValueError("Recon decay schedules require end <= start")
                if d.recon_schedule == "front_loaded" and d.recon_ratio_start >= 1:
                    raise ValueError(
                        "front_loaded mixed ratio must be < 1 to retain QA"
                    )
        if (
            d.max_question_tokens <= 0
            or d.max_conversation_tokens <= 0
            or d.max_answer_tokens <= 0
        ):
            raise ValueError(
                "question/conversation/answer token limits must be positive"
            )
        if d.sortish_bucket_multiplier <= 0:
            raise ValueError("sortish_bucket_multiplier must be positive")
        if d.max_qa_sample_per_context <= 0:
            raise ValueError("data.max_qa_sample_per_context must be positive")
        if d.question_padding_side not in {"left", "right"}:
            raise ValueError("data.question_padding_side must be left or right")
        if self.model.trainable_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(
                "model.trainable_dtype must be float32, float16 or bfloat16"
            )
        if self.model.lora_rank <= 0 or m.memory_length <= 0:
            raise ValueError("lora_rank and memory_length must be positive")
        if not m.context_layer_indices:
            raise ValueError("memory.context_layer_indices must not be empty")
        if any(type(i) is not int for i in m.context_layer_indices):
            raise ValueError("memory.context_layer_indices must contain integers")
        if len(set(m.context_layer_indices)) != len(m.context_layer_indices):
            raise ValueError("memory.context_layer_indices must select distinct layers")
        if m.memory_resampler_layers <= 0 or m.memory_resampler_heads <= 0:
            raise ValueError("memory resampler depth and heads must be positive")
        if m.memory_resampler_hidden_size <= 0 or m.memory_resampler_ffn_ratio <= 0:
            raise ValueError("memory resampler width and FFN ratio must be positive")
        if m.memory_resampler_hidden_size % m.memory_resampler_heads:
            raise ValueError(
                "memory_resampler_hidden_size must be divisible by memory_resampler_heads"
            )
        if m.memory_resampler_output_mode not in {"layer0", "layerwise"}:
            raise ValueError(
                "memory.memory_resampler_output_mode must be layer0 or layerwise"
            )
        if m.reader_memory_source not in {
            "memory_hidden",
            "memory_embedding",
            "layer_mean",
        }:
            raise ValueError(
                "memory.reader_memory_source must be memory_hidden, memory_embedding or layer_mean"
            )
        if loss.distill_weight < 0:
            raise ValueError("loss.distill_weight must be non-negative")
        if loss.distill_temperature <= 0:
            raise ValueError("loss.distill_temperature must be positive")
        if loss.distill_positions < 0:
            raise ValueError(
                "loss.distill_positions must be >= 0 (0 scores every position)"
            )
        if loss.distill_topk < 0:
            raise ValueError("loss.distill_topk must be >= 0")
        if loss.max_logits_rows <= 0:
            raise ValueError("loss.max_logits_rows must be positive")
        if m.readout_mode not in {
            "memory_question",
            "memory_only",
            "learned_prefix",
            "off",
        }:
            raise ValueError("invalid memory.readout_mode")
        if m.readout_length < 0:
            raise ValueError(
                "memory.readout_length must be >= 0 (0 disables the read-out)"
            )
        if m.readout_length > 0 and m.readout_mode in {
            "memory_question",
            "memory_only",
        }:
            if m.readout_layers <= 0:
                raise ValueError(
                    "memory.readout_layers must be positive when the read-out is on"
                )
            if m.readout_heads <= 0:
                raise ValueError(
                    "memory.readout_heads must be positive when the read-out is on"
                )
            if m.readout_hidden_size <= 0:
                raise ValueError("memory.readout_hidden_size must be positive")
            if m.readout_hidden_size % m.readout_heads:
                raise ValueError(
                    "memory.readout_hidden_size must be divisible by readout_heads"
                )
        if loss.qa_weight < 0 or loss.recon_weight < 0:
            raise ValueError("loss weights must be non-negative")
        qa_active = d.train_tasks in {"qa", "mix"}
        recon_active = d.train_tasks in {"recon", "mix"}
        if qa_active and loss.qa_weight <= 0:
            raise ValueError("QA training requires loss.qa_weight > 0")
        if recon_active and loss.recon_weight <= 0:
            raise ValueError("Recon training requires loss.recon_weight > 0")
        t = self.training
        if t.batch_size <= 0 or t.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if t.gradient_accumulation_steps <= 0:
            raise ValueError("training.gradient_accumulation_steps must be positive")
        if type(t.gradient_checkpoint_every_n_layers) is not int or t.gradient_checkpoint_every_n_layers < 0:
            raise ValueError("training.gradient_checkpoint_every_n_layers must be a non-negative integer")
        if t.max_grad_norm <= 0:
            raise ValueError("training.max_grad_norm must be positive")
        if self.evaluation.autoregressive_every <= 0:
            raise ValueError("evaluation.autoregressive_every must be positive")
        if self.evaluation.qa_batch_size <= 0:
            raise ValueError("evaluation.qa_batch_size must be positive")
        if self.evaluation.max_eval_contexts_per_dataset <= 0:
            raise ValueError(
                "evaluation.max_eval_contexts_per_dataset must be positive"
            )
        if self.logging.log_every <= 0:
            raise ValueError("logging.log_every must be positive")
        if self.logging.swanlab_mode not in {"online", "offline", "local", "disabled"}:
            raise ValueError(
                "logging.swanlab_mode must be online, offline, local or disabled"
            )


def dtype_from_name(name: str):
    import torch

    try:
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported torch dtype: {name}") from exc
