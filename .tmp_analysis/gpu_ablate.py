"""GPU ablation: where does the SQuAD F1 gap actually come from?

Conditions (all greedy, same SQuAD-dev examples, same trained checkpoint):
  mem_tf      current method, teacher-forced argmax through the 8-token memory prefix
  mem_ar      current method, autoregressive decoding through the memory prefix
  full_tf     memory bypassed, teacher-forced argmax with full [context,question,answer] attention
  full_ar     memory bypassed, autoregressive over [context, question]  (oracle upper bound)
  base_ar     LoRA disabled, full context, identical raw prompt format (no chat template)
  base_icl_ar LoRA disabled, full context, the ICL-baseline chat-template prompt

Each condition is scored twice:
  project  = src.metrics.qa_metrics(pred, answers[0])           (what the training runs log)
  official = src.icl_baseline.example_metrics(pred, all_refs)   (what the ICL baseline logs)

Usage (needs GPU device access):
  PYTHONPATH=. python .tmp_analysis/gpu_ablate.py [N] [MAX_NEW] [CKPT]
"""
from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ContextRecord, QARecord, collate_fn  # noqa: E402
from src.icl_baseline import ICLExample, example_metrics, render_prompt  # noqa: E402
from src.metrics import qa_metrics  # noqa: E402
from src.model import build_block_causal_mask, load_model  # noqa: E402
from utils.checkpoint import CheckpointManager  # noqa: E402
from utils.config import TrainConfig  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
MAX_NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 32
CKPT = sys.argv[3] if len(sys.argv) > 3 else "outputs/Qwen1.7B_20260916_024918/step-20000.pt"
CHUNK = 8
OUT = Path(".tmp_analysis/ablation_results.json")

cfg = TrainConfig.from_file("configs/qwen-1.7b/train.yaml")
cfg.validate()
device = torch.device("cuda:0")
tokenizer, model = load_model(cfg)
step, _ = CheckpointManager(Path(CKPT).parent).load(CKPT, model)
model.to(device).eval()
model = model.to(torch.bfloat16)
embed = model.qwen.get_input_embeddings()
print(f"ckpt={CKPT} step={step} N={N} chunk={CHUNK}", flush=True)


@contextmanager
def lora_disabled():
    mods = [m for m in model.modules() if hasattr(m, "scaling") and hasattr(m, "lora_A")]
    saved = [m.scaling for m in mods]
    for m in mods:
        m.scaling = 0.0
    try:
        yield
    finally:
        for m, s in zip(mods, saved):
            m.scaling = s


def build_examples(n):
    out = []
    with open("/data/lz/contexts/aggregated/squad/validation.jsonl", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            for qa in row.get("qa_pairs", []) or []:
                answers = qa.get("answers", qa.get("answer", []))
                answers = [str(a).strip() for a in (answers if isinstance(answers, list) else [answers]) if str(a).strip()]
                question = str(qa.get("question", "")).strip()
                if not question or not answers:
                    continue
                rec = ContextRecord(
                    str(row["context"]),
                    (QARecord(question, answers[0], "squad", str(qa.get("id", ""))),),
                    "squad", str(row.get("context_id", "")),
                )
                out.append((rec, answers, str(qa.get("split", ""))))
                break  # one QA per context keeps autoregressive cost bounded
            if len(out) >= n:
                break
    return out


def collate(chunk):
    return collate_fn(
        chunk, tokenizer, cfg.data.max_context_tokens, cfg.data.max_question_tokens,
        cfg.data.max_answer_tokens, cfg.data.append_eos, False, False,
        cfg.data.qa_per_context, False,
    )


examples = build_examples(N)
records = [e[0] for e in examples]
all_refs = [e[1] for e in examples]
splits = [e[2] for e in examples]
print("examples=%d  (v1.1=%d, v2.0=%d)" % (
    len(examples), sum("1.1" in s for s in splits), sum("2.0" in s for s in splits)), flush=True)

scores, raw = {}, {}


def score(name, preds):
    proj, offi, first = [], [], []
    for i, text in enumerate(preds):
        proj.append(qa_metrics(text, all_refs[i][0]))
        offi.append(example_metrics(text, all_refs[i]))
        head = text.strip().split()
        gold = all_refs[i][0].strip().split()
        first.append(float(bool(head and gold and head[0].lower() == gold[0].lower())))
    scores[name] = {
        "project_f1": sum(m["f1"] for m in proj) / len(proj),
        "project_em": sum(m["em"] for m in proj) / len(proj),
        "official_f1": sum(m["f1"] for m in offi) / len(offi),
        "official_em": sum(m["em"] for m in offi) / len(offi),
        "first_word_acc": sum(first) / len(first),
        "n": len(preds),
    }
    print("%-12s project F1=%.4f EM=%.4f | official F1=%.4f EM=%.4f | first-word acc=%.3f"
          % (name, scores[name]["project_f1"], scores[name]["project_em"],
             scores[name]["official_f1"], scores[name]["official_em"], scores[name]["first_word_acc"]), flush=True)
    raw[name] = preds
    OUT.write_text(json.dumps({"scores": scores, "predictions": raw, "refs": all_refs, "splits": splits},
                              ensure_ascii=False, indent=1))


tf_mem, ar_mem, tf_full, ar_full, base_ar, base_icl = [], [], [], [], [], []
chunks = [records[i:i + CHUNK] for i in range(0, len(records), CHUNK)]
chunk_refs = [all_refs[i:i + CHUNK] for i in range(0, len(records), CHUNK)]

with torch.no_grad():
    t0 = time.time()
    first_hit = {"mem": 0, "full": 0}
    for ci, (chunk, refs) in enumerate(zip(chunks, chunk_refs)):
        batch = collate(chunk)
        ids = {k: v.to(device) for k, v in batch.items() if k != "records"}
        ce, qe, ae = embed(ids["context_ids"]), embed(ids["question_ids"]), embed(ids["answer_ids"])
        prefix = model.encode_context_prefix(ce, ids["context_mask"])
        L, M = prefix.context_length, prefix.memory_length
        out = model.forward_qa_with_prefix(
            prefix, ids["qa_context_indices"], qe, ids["question_mask"], ae, ids["answer_mask"], ids["labels"])
        pred = out.logits.argmax(-1)[:, :-1]
        labels = out.labels[:, 1:]
        del out, prefix
        for i, rec in enumerate(batch["records"]):
            active = labels[i].ne(-100)
            tf_mem.append(tokenizer.decode(pred[i][active].tolist(), skip_special_tokens=True))
            first_hit["mem"] += int(pred[i][active][0].item() == labels[i][active][0].item())
        del pred, labels

        seq = torch.cat([ce, qe, ae], dim=1)
        mask = build_block_causal_mask(ids["context_mask"], 0, ids["question_mask"], ids["answer_mask"], seq.dtype)
        pos = torch.arange(seq.size(1), device=device).unsqueeze(0).expand(seq.size(0), -1)
        o = model.qwen(inputs_embeds=seq, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
        full_labels = torch.cat(
            [torch.full((seq.size(0), L + qe.size(1)), -100, dtype=ids["labels"].dtype, device=device),
             ids["labels"]], dim=1)[:, 1:]
        pred_full = o.logits.argmax(-1)[:, :-1]
        del o
        for i, rec in enumerate(batch["records"]):
            active = full_labels[i].ne(-100)
            tf_full.append(tokenizer.decode(pred_full[i][active].tolist(), skip_special_tokens=True))
            first_hit["full"] += int(pred_full[i][active][0].item() == full_labels[i][active][0].item())
        del pred_full, full_labels, mask, seq, pos, ce, qe, ae
        torch.cuda.empty_cache()
        print("  chunk %d/%d  L=%d  %.0fs" % (ci + 1, len(chunks), L, time.time() - t0), flush=True)
    print("exact first-answer-token accuracy  mem=%.3f  full=%.3f"
          % (first_hit["mem"] / len(records), first_hit["full"] / len(records)), flush=True)
    score("mem_tf", tf_mem)
    score("full_tf", tf_full)
    del tf_mem, tf_full
    torch.cuda.empty_cache()

    # ---------- autoregressive ----------
    t0 = time.time()
    for ci, (chunk, refs) in enumerate(zip(chunks, chunk_refs)):
        batch = collate(chunk)
        ids = {k: v.to(device) for k, v in batch.items() if k != "records"}
        ce, qe, ae = embed(ids["context_ids"]), embed(ids["question_ids"]), embed(ids["answer_ids"])
        prefix = model.encode_context_prefix(ce, ids["context_mask"])
        for i in range(len(chunk)):
            gen = model.generate_answer_with_prefix(
                prefix, ids["qa_context_indices"][i:i + 1], ids["question_ids"][i:i + 1],
                ids["question_mask"][i:i + 1], tokenizer, MAX_NEW)
            ar_mem.append(tokenizer.decode(gen[0].tolist(), skip_special_tokens=True))
        del prefix, ce, qe, ae, ids, batch
        torch.cuda.empty_cache()
    print("mem_ar %.0fs" % (time.time() - t0), flush=True)
    score("mem_ar", ar_mem)

    def gen_raw(ids_c, ids_q):
        full_ids = torch.cat([ids_c, ids_q]).unsqueeze(0)
        g = model.qwen.generate(
            input_ids=full_ids, attention_mask=torch.ones_like(full_ids), max_new_tokens=MAX_NEW,
            do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(g[0, full_ids.size(1):].tolist(), skip_special_tokens=True)

    t0 = time.time()
    for ci, chunk in enumerate(chunks):
        batch = collate(chunk)
        ids = {k: v.to(device) for k, v in batch.items() if k != "records"}
        for i in range(len(chunk)):
            ar_full.append(gen_raw(ids["context_ids"][i][ids["context_mask"][i]],
                                   ids["question_ids"][i][ids["question_mask"][i]]))
        del ids, batch
        torch.cuda.empty_cache()
    print("full_ar %.0fs" % (time.time() - t0), flush=True)
    score("full_ar", ar_full)

    with lora_disabled():
        t0 = time.time()
        for ci, chunk in enumerate(chunks):
            batch = collate(chunk)
            ids = {k: v.to(device) for k, v in batch.items() if k != "records"}
            for i in range(len(chunk)):
                base_ar.append(gen_raw(ids["context_ids"][i][ids["context_mask"][i]],
                                       ids["question_ids"][i][ids["question_mask"][i]]))
            del ids, batch
            torch.cuda.empty_cache()
        print("base_ar %.0fs" % (time.time() - t0), flush=True)
        score("base_ar", base_ar)

        t0 = time.time()
        for i, rec in enumerate(records):
            example = ICLExample(id=str(i), dataset="squad", context=rec.context,
                                 question=rec.qa_pairs[0].question, references=tuple(all_refs[i]))
            prompt = render_prompt(tokenizer, example, [], True)
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192,
                            add_special_tokens=False).to(device)
            g = model.qwen.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, use_cache=True,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(g[0, enc.input_ids.shape[1]:].tolist(), skip_special_tokens=True)
            base_icl.append(text.splitlines()[0].strip() if text.strip() else "")
        print("base_icl_ar %.0fs" % (time.time() - t0), flush=True)
        score("base_icl_ar", base_icl)

print("=" * 90)
for k, v in scores.items():
    print("%-12s projectF1=%.4f officialF1=%.4f firstWord=%.3f" % (k, v["project_f1"], v["official_f1"], v["first_word_acc"]))
