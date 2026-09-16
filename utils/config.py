from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import re
from typing import Any

import yaml


# ``${VAR}`` or ``${VAR:-default}`` inside a configuration string.
ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: str) -> str:
    """Expand ``${VAR}`` and ``${VAR:-default}`` references in one configuration string.

    Machine-specific locations must not be baked into tracked files. The model directory and
    the aggregated data root live somewhere different on every cluster, and editing the YAML
    there turns every ``git pull`` into a conflict -- which is exactly what happened. Writing
    them as ``${MODEL_ROOT:-/data/lz/hf_cache/hub}`` keeps the default valid where it was
    written and lets another machine override it through the environment (the usual route is
    a gitignored ``scripts/env.local.sh``, see the README).

    A bare ``${VAR}`` with no fallback raises when the variable is unset, so a typo cannot
    silently turn into an empty path.
    """
    def replace(match: re.Match) -> str:
        name, default = match.group(1), match.group(2)
        if os.environ.get(name):
            return os.environ[name]
        if default is not None:
            return default
        raise ValueError(
            f"${{{name}}} is not set and has no default; export {name} or write "
            f"${{{name}:-<default>}} in the configuration"
        )

    return ENV_REFERENCE.sub(replace, value)


def expand_env_values(value: Any) -> Any:
    """Apply :func:`expand_env` to every string in a nested configuration structure."""
    if isinstance(value, str):
        return expand_env(value)
    if isinstance(value, dict):
        return {key: expand_env_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [expand_env_values(item) for item in value]
    return value


@dataclass
class ModelConfig:
    name_or_path: str = (
        "/data/lz/hf_cache/hub/models--Qwen--Qwen3-1.7B/"
        "snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    )
    torch_dtype: str = "bfloat16"
    # Dtype of the trainable pieces (memory tokens, LoRA, reconstruction decoders).
    # Kept separate from torch_dtype so that AdamW keeps float32 parameters and
    # moments while the frozen backbone stays at its checkpoint dtype.
    trainable_dtype: str = "float32"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    # LoRA implementation. "false" (default) uses the in-repo StaticLoRALinear,
    # which disables autocast at the module boundary so float32 adapters really
    # compute in float32. "true" uses PEFT instead; it is opt-in because it is a
    # second implementation that needs its own autocast handling
    # (see src/model.py:disable_autocast_for_peft_lora) and is not covered by the
    # same tests. Leaving it unset keeps a run independent of whether peft is
    # installed.
    use_peft: bool = False
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


@dataclass
class MemoryConfig:
    memory_length: int = 8
    decoder_hidden_size: int = 256
    decoder_ffn_ratio: int = 2
    decoder_heads: int = 8
    reconstruction_weight: float = 1.0
    qa_weight: float = 1.0
    reconstruction_loss: str = "mse_cosine"
    reconstruction_cosine_weight: float = 0.1
    # How the shared context_lm unembedding is parameterised.
    #   "linear" (default): a from-scratch [vocab, D] weight matrix -- the original
    #                       behaviour, kept as the default so nothing changes silently.
    #   "tied":             W = E @ adapter, with E the frozen (tied) input embedding of
    #                       the backbone. Trains D*H ~= 0.5M instead of D*vocab ~= 39M
    #                       and starts from the pretrained token geometry.
    head_mode: str = "linear"
    # Initialisation of the tied adapter (ignored in "linear" mode):
    #   "auto"              -> "memory_projection" for tied, "random" otherwise
    #   "random"            -> default nn.Linear init
    #   "memory_projection" -> up-projection through the decoders' memory projection, so
    #                          step-0 logits already score the tokens the memory points at
    head_init: str = "auto"
    # context_lm: number of context positions scored per context per step. The shared
    # vocabulary head is applied to num_layers * context_lm_positions rows, so this is
    # the knob that bounds the auxiliary objective's cost. <= 0 means "all positions".
    context_lm_positions: int = 256
    contrastive_weight: float = 0.0
    contrastive_temperature: float = 0.07
    contrastive_margin: float = 1.0


@dataclass
class DataConfig:
    root: str = "/data/lz/contexts/aggregated"
    dataset: str = "all"
    train_datasets: tuple[str, ...] | None = None
    validation_datasets: tuple[str, ...] | None = None
    test_datasets: tuple[str, ...] | None = None
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    max_context_tokens: int = 2048
    max_question_tokens: int = 128
    max_answer_tokens: int = 128
    max_samples: int | None = None
    train_max_samples: int | None = None
    validation_max_samples: int | None = None
    filter_long_context: bool = True
    filter_no_qa: bool = True
    sortish_bucket_multiplier: int = 50
    cache_sortish_lengths: bool = True
    cache_dataset: bool = True
    append_eos: bool = True
    use_chat_template: bool = False
    chat_template_enable_thinking: bool = False
    qa_per_context: int = 4
    # Left-padding keeps the question's real tokens adjacent to the answer; right
    # padding inserts pad positions between them, so the first answer token would be
    # supervised from a padding hidden state.  Set to "right" only to reproduce old runs.
    question_padding_side: str = "left"
    # "append" writes EOS after the last answer token; the legacy "overwrite" mode
    # overwrites that token whenever the answer is the longest in the batch.
    eos_mode: str = "append"


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
    teacher_forced_every: int = 2000
    autoregressive_every: int = 4000
    max_new_tokens: int = 128
    qa_batch_size: int = 4
    # QA rows each rank decodes during an autoregressive evaluation. Decoding is far
    # more expensive than a teacher-forced forward, and a rank that spends longer than
    # the NCCL watchdog timeout (10 minutes) inside an evaluation makes the other ranks
    # abort with "Watchdog caught collective operation timeout". The global number of
    # decoded rows is autoregressive_max_qa * world_size.
    autoregressive_max_qa: int = 256


@dataclass
class TrainingConfig:
    batch_size: int = 2
    epochs: int = 1
    max_grad_norm: float = 1.0
    seed: int = 42


@dataclass
class CheckpointConfig:
    output_dir: str = "outputs/qwen-metalora"
    save_every_steps: int = 1000


@dataclass
class LoggingConfig:
    wandb_project: str = "MemoryQwen"
    wandb_run_name: str | None = None
    wandb_mode: str = "online"
    wandb_run_id: str | None = None


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

    @classmethod
    def from_file(cls, path: str | Path) -> "TrainConfig":
        path = Path(path)
        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(f"configuration must be a YAML file (.yaml/.yml), got: {path}")
        text = path.read_text(encoding="utf-8")
        values = expand_env_values(yaml.safe_load(text) or {})
        values = dict(values)
        data_values = dict(values.get("data", {}))
        for key in ("train_datasets", "validation_datasets", "test_datasets"):
            if isinstance(data_values.get(key), list):
                data_values[key] = tuple(data_values[key])
        optimizer_values = dict(values.get("optimizer", {}))
        if isinstance(optimizer_values.get("betas"), list):
            optimizer_values["betas"] = tuple(optimizer_values["betas"])
        model_values = dict(values.get("model", {}))
        if isinstance(model_values.get("target_modules"), list):
            model_values["target_modules"] = tuple(model_values["target_modules"])
        return cls(
            model=ModelConfig(**model_values),
            memory=MemoryConfig(**values.get("memory", {})),
            data=DataConfig(**data_values),
            optimizer=OptimizerConfig(**optimizer_values),
            scheduler=SchedulerConfig(**values.get("scheduler", {})),
            evaluation=EvaluationConfig(**values.get("evaluation", {})),
            training=TrainingConfig(**values.get("training", {})),
            checkpoint=CheckpointConfig(**values.get("checkpoint", {})),
            logging=LoggingConfig(**values.get("logging", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        d, m = self.data, self.memory
        if not 0 < d.max_context_tokens <= 2048:
            raise ValueError("data.max_context_tokens must be in (0, 2048]")
        if d.max_question_tokens <= 0 or d.max_answer_tokens <= 0:
            raise ValueError("question/answer token limits must be positive")
        if d.sortish_bucket_multiplier <= 0:
            raise ValueError("sortish_bucket_multiplier must be positive")
        if d.qa_per_context <= 0:
            raise ValueError("data.qa_per_context must be positive")
        if d.question_padding_side not in {"left", "right"}:
            raise ValueError("data.question_padding_side must be left or right")
        if d.eos_mode not in {"overwrite", "append"}:
            raise ValueError("data.eos_mode must be overwrite or append")
        if self.model.trainable_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("model.trainable_dtype must be float32, float16 or bfloat16")
        if self.model.lora_rank <= 0 or m.memory_length <= 0:
            raise ValueError("lora_rank and memory_length must be positive")
        if m.decoder_hidden_size <= 0 or m.decoder_ffn_ratio <= 0 or m.decoder_heads <= 0:
            raise ValueError("decoder_hidden_size, decoder_ffn_ratio and decoder_heads must be positive")
        if m.decoder_hidden_size % m.decoder_heads:
            raise ValueError("memory.decoder_hidden_size must be divisible by memory.decoder_heads")
        if m.reconstruction_loss not in {"mse", "cosine", "mse_cosine", "context_lm"}:
            raise ValueError("reconstruction_loss must be mse, cosine, mse_cosine, or context_lm")
        if m.head_mode not in {"linear", "tied"}:
            raise ValueError("memory.head_mode must be linear or tied")
        if m.head_init not in {"auto", "random", "memory_projection"}:
            raise ValueError("memory.head_init must be auto, random or memory_projection")
        if m.context_lm_positions < 0:
            raise ValueError("memory.context_lm_positions must be >= 0 (0 scores every position)")
        if m.qa_weight < 0 or m.reconstruction_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if m.qa_weight == 0 and m.reconstruction_weight == 0:
            raise ValueError("at least one main loss weight must be positive")
        t = self.training
        if t.batch_size <= 0 or t.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if t.max_grad_norm <= 0:
            raise ValueError("training.max_grad_norm must be positive")
        if self.checkpoint.save_every_steps <= 0:
            raise ValueError("checkpoint.save_every_steps must be positive")
        if self.evaluation.teacher_forced_every <= 0 or self.evaluation.autoregressive_every <= 0:
            raise ValueError("evaluation intervals must be positive")
        if self.evaluation.qa_batch_size <= 0:
            raise ValueError("evaluation.qa_batch_size must be positive")
        if self.evaluation.autoregressive_max_qa <= 0:
            raise ValueError("evaluation.autoregressive_max_qa must be positive")


def dtype_from_name(name: str):
    import torch
    try:
        return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported torch dtype: {name}") from exc
