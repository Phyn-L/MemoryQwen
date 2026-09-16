"""CPU smoke test for the two-dtype (bf16 backbone / fp32 trainable) design.

Checks, for both trainable dtypes:
  * frozen backbone stays bfloat16, trainable parameters take the requested dtype;
  * forward/backward under a bfloat16 autocast context produces finite gradients in the
    trainable dtype;
  * one AdamW step actually changes a trainable parameter (this is what bfloat16
    quantisation destroys at lr=1e-4);
  * the reconstruction loss compares float32 predictions against bfloat16 targets;
  * generation still runs.

Usage: python .tmp_analysis/verify_dtype.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ContextRecord, QARecord, collate_fn  # noqa: E402
from src.losses import qa_loss, reconstruction_loss  # noqa: E402
from src.model import MetaLoRA  # noqa: E402

TOKENIZER = ("/data/lz/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/"
             "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)

rows = [
    ContextRecord("Alpha beta gamma delta epsilon zeta eta theta.", (QARecord("Which Greek letter is third?", "gamma", "x", ""),), "x", ""),
    ContextRecord("One two three four five six seven eight nine ten eleven.", (QARecord("What comes right after six in that list?", "seven", "x", ""),), "x", ""),
]
batch = collate_fn(
    rows, tokenizer,
    max_context_tokens=128, max_question_tokens=32, max_answer_tokens=16,
    append_eos=True, use_chat_template=False, chat_template_enable_thinking=False,
    qa_per_context=2, sample_qa=False,
    question_padding_side="left", eos_mode="append",
)


def build(trainable_dtype):
    cfg = Qwen3Config(vocab_size=152064, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, max_position_embeddings=256, dtype=torch.bfloat16)
    qwen = Qwen3ForCausalLM(cfg).to(torch.bfloat16)
    model = MetaLoRA(
        qwen, rank=4, alpha=8.0, memory_length=8, decoder_hidden_size=32, decoder_heads=4,
        decoder_ffn_ratio=2,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        max_context_tokens=128, trainable_dtype=trainable_dtype,
    )
    for name, p in model.named_parameters():
        p.requires_grad = ("lora_A" in name or "lora_B" in name or name == "memory_tokens"
                           or name.startswith("decoders."))
    return model


for trainable_dtype in (torch.float32, torch.bfloat16):
    print("=" * 90)
    print("trainable_dtype =", trainable_dtype)
    model = build(trainable_dtype)
    model.train()
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n: p for n, p in model.named_parameters() if not p.requires_grad}
    print("  trainable dtypes:", model.trainable_parameter_dtypes(),
          " params:", len(trainable))
    print("  frozen backbone dtypes:",
          {str(p.dtype) for p in frozen.values()})
    assert all(p.dtype == trainable_dtype for p in trainable.values())
    assert all(p.dtype == torch.bfloat16 for p in frozen.values())

    embed = model.qwen.get_input_embeddings()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = model(
            embed(batch["context_ids"]), batch["context_mask"],
            embed(batch["question_ids"]), batch["question_mask"],
            embed(batch["answer_ids"]), batch["answer_mask"],
            batch["labels"], batch["qa_context_indices"],
        )
        qa = qa_loss(out.logits, out.labels)
        reconstruction = reconstruction_loss(
            out.reconstruction, out.context_target, out.context_mask, 0.5, "mse_cosine")
        total = qa + reconstruction
    total.backward()
    bad = [n for n, p in trainable.items()
           if p.grad is None or not torch.isfinite(p.grad).all()
           or p.grad.dtype != trainable_dtype]
    print("  qa=%.4f recon=%.4f (recon pred dtype %s, target dtype %s)"
          % (float(qa), float(reconstruction), out.reconstruction.dtype, out.context_target.dtype))
    print("  trainable params with bad grads:", len(bad), bad[:3])

    # does one AdamW step at lr=1e-4 actually move the parameters?
    optimizer = torch.optim.AdamW([p for p in trainable.values()], lr=1e-4)
    before = {n: p.detach().clone() for n, p in trainable.items()}
    optimizer.step()
    moved = sum(1 for n, p in trainable.items() if not torch.equal(p.detach(), before[n]))
    print("  parameters changed by one AdamW step at lr=1e-4: %d/%d" % (moved, len(trainable)))

    model.eval()
    with torch.no_grad():
        prefix = model.encode_context_prefix(embed(batch["context_ids"]), batch["context_mask"])
        generated = model.generate_answer_with_prefix(
            prefix, batch["qa_context_indices"], batch["question_ids"],
            batch["question_mask"], tokenizer, 4)
    print("  generation ok, shape", tuple(generated.shape))
print("=" * 90)
print("DTYPE SMOKE OK")
