from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    name_or_path: str = (
        "/data/lz/hf_cache/hub/models--Qwen--Qwen3-1.7B/"
        "snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    )
    torch_dtype: str = "bfloat16"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    freeze_backbone: bool = True
    gradient_checkpointing: bool = False


@dataclass
class MemoryConfig:
    num_memory_tokens: int = 8
    encoder_heads: int = 8
    decoder_heads: int = 8
    reconstruction_weight: float = 1.0
    qa_weight: float = 1.0
    reconstruction_loss: str = "mse_cosine"
    reconstruction_cosine_weight: float = 0.1
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
    test_max_samples: int | None = None
    filter_long_context: bool = True
    filter_no_qa: bool = True
    sortish_bucket_multiplier: int = 50
    append_eos: bool = True
    use_chat_template: bool = False
    chat_template_enable_thinking: bool = False


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


@dataclass
class TrainingConfig:
    batch_size: int = 2
    epochs: int = 1
    grad_accumulation: int = 1
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
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
                values = yaml.safe_load(text) or {}
            except ImportError as exc:
                raise RuntimeError("PyYAML is required to read YAML configs; use configs/default.json without optional dependencies") from exc
        else:
            values = json.loads(text)
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
        section_names = {
            "model", "memory", "data", "optimizer",
            "scheduler", "evaluation", "training", "checkpoint", "logging",
        }
        top = {k: v for k, v in values.items() if k not in section_names}

        # Accept legacy flat configs while serializing all new configs by section.
        training_values = dict(values.get("training", {}))
        for key in ("batch_size", "epochs", "grad_accumulation", "max_grad_norm", "seed"):
            if key in top:
                training_values.setdefault(key, top.pop(key))
        checkpoint_values = dict(values.get("checkpoint", {}))
        for key in ("output_dir", "save_every_steps"):
            if key in top:
                checkpoint_values.setdefault(key, top.pop(key))
        logging_values = dict(values.get("logging", {}))
        for key in ("wandb_project", "wandb_run_name", "wandb_mode", "wandb_run_id"):
            if key in top:
                logging_values.setdefault(key, top.pop(key))
        return cls(
            model=ModelConfig(**model_values),
            memory=MemoryConfig(**values.get("memory", {})),
            data=DataConfig(**data_values),
            optimizer=OptimizerConfig(**optimizer_values),
            scheduler=SchedulerConfig(**values.get("scheduler", {})),
            evaluation=EvaluationConfig(**values.get("evaluation", {})),
            training=TrainingConfig(**training_values),
            checkpoint=CheckpointConfig(**checkpoint_values),
            logging=LoggingConfig(**logging_values),
            **top,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=list), encoding="utf-8")

    def validate(self) -> None:
        d, m = self.data, self.memory
        if not 0 < d.max_context_tokens <= 2048:
            raise ValueError("data.max_context_tokens must be in (0, 2048]")
        if d.max_question_tokens <= 0 or d.max_answer_tokens <= 0:
            raise ValueError("question/answer token limits must be positive")
        if d.sortish_bucket_multiplier <= 0:
            raise ValueError("sortish_bucket_multiplier must be positive")
        if self.model.lora_rank <= 0 or m.num_memory_tokens <= 0:
            raise ValueError("lora_rank and num_memory_tokens must be positive")
        if m.reconstruction_loss not in {"mse", "cosine", "mse_cosine"}:
            raise ValueError("reconstruction_loss must be mse, cosine, or mse_cosine")
        if m.qa_weight < 0 or m.reconstruction_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if m.qa_weight == 0 and m.reconstruction_weight == 0:
            raise ValueError("at least one main loss weight must be positive")
        t = self.training
        if t.batch_size <= 0 or t.epochs <= 0 or t.grad_accumulation <= 0:
            raise ValueError("batch_size, epochs and grad_accumulation must be positive")
        if t.max_grad_norm <= 0:
            raise ValueError("training.max_grad_norm must be positive")
        if self.checkpoint.save_every_steps <= 0:
            raise ValueError("checkpoint.save_every_steps must be positive")
        if self.evaluation.teacher_forced_every <= 0 or self.evaluation.autoregressive_every <= 0:
            raise ValueError("evaluation intervals must be positive")


def dtype_from_name(name: str):
    import torch
    try:
        return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported torch dtype: {name}") from exc
