"""Generate the 4 arm configs for the parallel ablation."""
import pathlib

BASE = """model:
  name_or_path: /data/lz/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
  torch_dtype: bfloat16
  lora_rank: 8
  lora_alpha: 16.0
  lora_dropout: 0.0
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
memory:
  memory_length: {memory_length}
  encoder_heads: 8
  decoder_hidden_size: 256
  decoder_ffn_ratio: 2
  decoder_heads: 8
  reconstruction_weight: {recon_weight}
  qa_weight: 1.0
  reconstruction_loss: mse_cosine
  reconstruction_cosine_weight: 0.5
  contrastive_weight: 0.0
data:
  root: /data/lz/contexts/aggregated
  train_datasets: [squad]
  validation_datasets: [squad]
  test_datasets: [squad]
  max_context_tokens: 2048
  max_question_tokens: 128
  max_answer_tokens: 128
  train_max_samples: {train_max_samples}
  validation_max_samples: 400
  test_max_samples: 300
  filter_long_context: true
  filter_no_qa: true
  sortish_bucket_multiplier: 50
  cache_sortish_lengths: true
  cache_dataset: true
  use_chat_template: false
  chat_template_enable_thinking: false
  qa_per_context: 4
  question_padding_side: {pad}
  eos_mode: {eos_mode}
evaluation:
  teacher_forced_every: 1000
  autoregressive_every: 1000000
  max_new_tokens: 32
  qa_batch_size: 4
training:
  batch_size: 1
  epochs: 1
  grad_accumulation: 1
  max_grad_norm: 1.0
  seed: 42
optimizer:
  name: adamw
  lr: 1.0e-4
  weight_decay: 0.01
scheduler:
  name: cosine
  warmup_steps: 300
checkpoint:
  output_dir: outputs/{arm}
  save_every_steps: 100000
logging:
  wandb_project: MemoryQwen
  wandb_mode: disabled
"""

ARMS = {
    "armA_control":        dict(memory_length=8, recon_weight=1.0, pad="right", eos_mode="overwrite", train_max_samples=6000),
    "armB_fixed":          dict(memory_length=8, recon_weight=1.0, pad="left", eos_mode="append", train_max_samples=6000),
    "armC_fixed_mem64":    dict(memory_length=64, recon_weight=1.0, pad="left", eos_mode="append", train_max_samples=6000),
    "armD_fixed_norecon":  dict(memory_length=8, recon_weight=0.0, pad="left", eos_mode="append", train_max_samples=6000),
}

out = pathlib.Path(__file__).resolve().parent
(out / "configs").mkdir(exist_ok=True)
for arm, options in ARMS.items():
    (out / "configs" / f"{arm}.yaml").write_text(BASE.format(arm=arm, **options))
    (out / "run" / arm).mkdir(parents=True, exist_ok=True)
    print("wrote", arm, options)
