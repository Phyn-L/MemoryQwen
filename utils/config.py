from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from .machines import fill_missing, machine_environ, resolve_machine

# ``${VAR}`` or ``${VAR:-default}`` inside a configuration string.
ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge nested configuration mappings; scalar and list values are replaced."""
    merged = dict(base)
    for key, value in override.items():
        if (
            isinstance(merged.get(key), Mapping)
            and isinstance(value, Mapping)
        ):
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
    """Accept grouped YAML data settings while keeping the flat DataConfig API."""
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        if key == "validation_max_samples_per_dataset" and isinstance(value, Mapping):
            flattened[key] = dict(value)
            continue
        if isinstance(value, Mapping):
            nested = _flatten_data_groups(value)
            duplicate = set(flattened).intersection(nested)
            if duplicate:
                raise ValueError(f"duplicate data fields after grouping: {sorted(duplicate)}")
            flattened.update(nested)
        else:
            if key in flattened:
                raise ValueError(f"duplicate data field: {key}")
            flattened[key] = value
    return flattened


def expand_env(value: str, overrides: Mapping[str, str] | None = None) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}`` references in one configuration string.

    Machine-specific locations must not be baked into tracked files. The model directory and
    the aggregated data root live somewhere different on every cluster, and editing the YAML
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
class DataConfig:
    root: str = "/data/lz/contexts"
    conversation_file: str | None = "/data/lz/contexts/ift_c1qa.jsonl"
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
    validation_max_samples: int | None = None
    validation_source_file: str | None = None
    validation_output_dir: str | None = None
    validation_max_context_tokens: int | None = None
    validation_max_samples_per_dataset: dict[str, int] | None = None
    validation_sampling_workers: int = 4
    filter_long_context: bool = True
    filter_no_qa: bool = True
    tokenizer_filter_batch_size: int = 2048
    sortish_bucket_multiplier: int = 50
    cache_sortish_lengths: bool = True
    cache_offsets: bool = True
    append_eos: bool = True
    chat_template_enable_thinking: bool = False
    qa_per_context: int = 4
    # Left-padding keeps the question's real tokens adjacent to the answer; right
    # padding inserts pad positions between them, so the first answer token would be
    # supervised from a padding hidden state.  Set to "right" only to reproduce old runs.
    question_padding_side: str = "left"
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    # "append" writes EOS after the last answer token; the legacy "overwrite" mode
    # overwrites that token whenever the answer is the longest in the batch.


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
    # QA rows the *whole* autoregressive evaluation decodes, split into contiguous windows
    # of the validation split's row order (src/evaluator.py::row_window). A global budget
    # rather than a per-rank one, so changing the GPU count changes neither the scored rows
    # nor the noise level; per-rank work still shrinks as ranks grow, which is what keeps the
    # evaluation inside the NCCL watchdog timeout (10 minutes).
    evaluation_max_samples: int = 1280


@dataclass
class TrainingConfig:
    batch_size: int = 2
    epochs: int = 1
    max_grad_norm: float = 1.0
    seed: int = 42
    deterministic: bool = False


@dataclass
class CheckpointConfig:
    output_dir: str = "outputs/qwen-metalora"
    init_from: str | None = None


@dataclass
class LoggingConfig:
    swanlab_project: str = "MemoryQwen"
    # W&B name and checkpoint subdirectory; empty means model name + timestamp.
    swanlab_run_name: str | None = None
    swanlab_mode: str = "online"
    swanlab_run_id: str | None = None
    # Cadence of the training-loss scalars (``train/*``), in optimizer steps. The two
    # evaluation sections keep their own cadence, so this only controls how densely the
    # loss curves are sampled. 10 is the historical value; a single-epoch run can afford
    # to be denser (or sparser) without touching any other schedule.
    log_every: int = 10

    @property
    def wandb_run_name(self):
        return self.swanlab_run_name


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    # Which entry of ``utils/machines.py::MACHINES`` supplied the paths (None = the config's
    # own defaults). Recorded so a run's swanlab config and checkpoint say where it ran.
    machine: str | None = None

    @classmethod
    def from_file(cls, path: str | Path, machine: str | None = None) -> "TrainConfig":
        """Load a config, resolving machine-specific paths.

        ``machine`` (or ``machine:`` in the file, or ``MACHINE`` in the environment) selects
        an entry from ``utils/machines.py::MACHINES``; a machine that was named outright
        supplies ``MODEL_ROOT``/``DATA_ROOT``/``SWANLAB_MODE`` for this load, while one only
        recognised from the hostname just fills in what the environment did not set.
        """
        path = Path(path)
        from utils.config_paths import resolve_config_path

        path = resolve_config_path(path)
        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(
                f"configuration must be a YAML file (.yaml/.yml), got: {path}"
            )
        raw = _load_yaml_with_base(path)
        return cls.from_dict(raw, machine=machine)

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, Any], machine: str | None = None
    ) -> "TrainConfig":
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
        # Conversation files are the only supported training source. Keep the default
        # machine-aware so a legacy YAML without an explicit path still resolves to the
        # current JSONL location.
        data_values.setdefault(
            "conversation_file",
            f"{overrides.get('DATA_ROOT', '/data/lz/contexts')}/ift_c1qa.jsonl",
        )
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
        if isinstance(memory_values.get("context_layer_indices"), list):
            memory_values["context_layer_indices"] = tuple(
                memory_values["context_layer_indices"]
            )
        checkpoint_values = dict(values.get("checkpoint", {}))
        # Legacy checkpoints embed this removed scheduling field.
        checkpoint_values.pop("save_every_steps", None)
        return cls(
            model=ModelConfig(**model_values),
            memory=MemoryConfig(**memory_values),
            data=DataConfig(**data_values),
            optimizer=OptimizerConfig(**optimizer_values),
            scheduler=SchedulerConfig(**values.get("scheduler", {})),
            evaluation=EvaluationConfig(**values.get("evaluation", {})),
            training=TrainingConfig(**values.get("training", {})),
            checkpoint=CheckpointConfig(**checkpoint_values),
            logging=LoggingConfig(**values.get("logging", {})),
            machine=resolved,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        d, m = self.data, self.memory
        if self.model.attn_implementation not in {"sdpa", "eager"}:
            raise ValueError("model.attn_implementation must be sdpa or eager")
        if d.max_context_tokens <= 0:
            raise ValueError("data.max_context_tokens must be positive")
        if not (
            d.conversation_file
            or d.train_conversation_file
            or d.validation_conversation_file
            or d.test_conversation_file
        ):
            raise ValueError(
                "conversation data requires at least one conversation file"
            )
        if d.max_input_tokens <= 0:
            raise ValueError("data.max_input_tokens must be positive")
        if d.max_input_tokens < d.max_context_tokens:
            raise ValueError(
                "data.max_input_tokens must be >= data.max_context_tokens"
            )
        if d.tokenizer_filter_batch_size <= 0:
            raise ValueError("data.tokenizer_filter_batch_size must be positive")
        if d.num_workers < 0:
            raise ValueError("data.num_workers must be non-negative")
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
            raise ValueError("question/conversation/answer token limits must be positive")
        if d.sortish_bucket_multiplier <= 0:
            raise ValueError("sortish_bucket_multiplier must be positive")
        if d.qa_per_context <= 0:
            raise ValueError("data.qa_per_context must be positive")
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
        if m.distill_weight < 0:
            raise ValueError("memory.distill_weight must be non-negative")
        if m.distill_temperature <= 0:
            raise ValueError("memory.distill_temperature must be positive")
        if m.distill_positions < 0:
            raise ValueError(
                "memory.distill_positions must be >= 0 (0 scores every position)"
            )
        if m.distill_topk < 0:
            raise ValueError("memory.distill_topk must be >= 0")
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
        if m.qa_weight < 0 or m.recon_weight < 0:
            raise ValueError("loss weights must be non-negative")
        qa_active = d.train_tasks in {"qa", "mix"}
        recon_active = d.train_tasks in {"recon", "mix"}
        if qa_active and m.qa_weight <= 0:
            raise ValueError("QA training requires memory.qa_weight > 0")
        if recon_active and m.recon_weight <= 0:
            raise ValueError("Recon training requires memory.recon_weight > 0")
        t = self.training
        if t.batch_size <= 0 or t.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if t.max_grad_norm <= 0:
            raise ValueError("training.max_grad_norm must be positive")
        if self.evaluation.autoregressive_every <= 0:
            raise ValueError("evaluation.autoregressive_every must be positive")
        if self.evaluation.qa_batch_size <= 0:
            raise ValueError("evaluation.qa_batch_size must be positive")
        if self.evaluation.evaluation_max_samples <= 0:
            raise ValueError("evaluation.evaluation_max_samples must be positive")
        if self.logging.log_every <= 0:
            raise ValueError("logging.log_every must be positive")


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
