from __future__ import annotations
import json, os, random
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import Dataset, Sampler
import torch
from tqdm.auto import tqdm


def _progress_enabled() -> bool:
    return os.environ.get("RANK", "0") in {"0", "-1"}

@dataclass
class Record:
    context: str; question: str; answer: str; dataset: str = ""; context_id: str = ""

def render_question(tokenizer, question: str, use_chat_template: bool = False, enable_thinking: bool = False) -> str:
    if not use_chat_template or not getattr(tokenizer, "apply_chat_template", None):
        return question
    messages = [{"role": "user", "content": question}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(messages, **kwargs)

def _answer(qa):
    a = qa.get("answers", qa.get("answer", [])); a = a if isinstance(a, list) else [a]
    return next((str(x).strip() for x in a if x and str(x).strip()), None)

class AggregatedQADataset(Dataset):
    def __init__(self, root, dataset="all", split="train", tokenizer=None, max_context_tokens=2048, max_samples=None, filter_long_context=True, filter_no_qa=True, allow_empty=False):
        self.records=[]; root=Path(root)
        if dataset is None: dataset="all"
        names=[p.name for p in sorted(root.iterdir()) if p.is_dir()] if dataset=="all" else ([dataset] if isinstance(dataset,str) else list(dataset))
        files = [(name, root / name / f"{split}.jsonl") for name in names]
        files = [(name, path) for name, path in files if path.exists()]
        with tqdm(files, total=len(files), desc=f"Loading {split} data", unit="file", disable=not _progress_enabled()) as file_bar:
          for name, path in file_bar:
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                row=json.loads(line); context=str(row.get("context",""))
                if filter_long_context and tokenizer is not None:
                    context_ids = tokenizer(
                        context,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=max_context_tokens + 1,
                    ).input_ids
                    if len(context_ids) > max_context_tokens:
                        continue
                for qa in row.get("qa_pairs", []) or []:
                    ans=_answer(qa)
                    if filter_no_qa and (not qa.get("question") or not ans): continue
                    question = str(qa["question"]).strip()
                    if ans: self.records.append(Record(context, question, ans, name, str(row.get("context_id",""))))
                    if max_samples is not None and len(self.records)>=max_samples: break
                if max_samples is not None and len(self.records)>=max_samples: break
                file_bar.set_postfix(records=len(self.records))
            if max_samples is not None and len(self.records)>=max_samples: break
        if not self.records and not allow_empty: raise RuntimeError(f"No usable records found under {root} split={split} datasets={names}")
    def __len__(self): return len(self.records)
    def __getitem__(self, i): return self.records[i]

class SortishSampler(Sampler[int]):
    def __init__(self, dataset, tokenizer, batch_size, bucket_multiplier=50, seed=42, use_chat_template=False, chat_template_enable_thinking=False):
        self.dataset,self.batch_size,self.seed=dataset,batch_size,seed; self.epoch=0
        lengths=[]
        with tqdm(total=len(dataset.records), desc="Preparing sortish lengths", unit="sample", disable=not _progress_enabled()) as length_bar:
            for record in dataset.records:
                question = render_question(tokenizer, record.question, use_chat_template, chat_template_enable_thinking)
                lengths.append(len(tokenizer(record.context,add_special_tokens=False).input_ids)+len(tokenizer(question,add_special_tokens=False).input_ids))
                length_bar.update(1)
        self.lengths=lengths; self.bucket_multiplier=bucket_multiplier; self._rebuild()
    def _rebuild(self):
        self.buckets=[]; size=max(self.batch_size, self.batch_size*self.bucket_multiplier)
        order=sorted(range(len(self.lengths)), key=self.lengths.__getitem__)
        for i in range(0,len(order),size):
            b=order[i:i+size]; random.Random(self.seed+i+self.epoch).shuffle(b); self.buckets.extend([b[j:j+self.batch_size] for j in range(0,len(b),self.batch_size)])
    def __iter__(self):
        g=random.Random(self.seed+self.epoch); batches=list(self.buckets); g.shuffle(batches); return iter([i for b in batches for i in b])
    def __len__(self): return len(self.dataset)
    def set_epoch(self,e): self.epoch=e; self._rebuild()

def collate_fn(rows, tokenizer, max_context_tokens=2048, max_question_tokens=128, max_answer_tokens=128, append_eos=True, use_chat_template=False, chat_template_enable_thinking=False):
    contexts=[r.context for r in rows]; questions=[render_question(tokenizer, r.question, use_chat_template, chat_template_enable_thinking) for r in rows]; answers=[r.answer for r in rows]
    def enc(xs,n): return tokenizer(xs,return_tensors="pt",padding=True,truncation=True,max_length=n,add_special_tokens=False)
    c=enc(contexts,max_context_tokens); q=enc(questions,max_question_tokens); a=enc(answers,max_answer_tokens)
    if append_eos and getattr(tokenizer, "eos_token_id", None) is not None:
        eos=int(tokenizer.eos_token_id)
        for i in range(a.input_ids.size(0)):
            valid=int(a.attention_mask[i].sum())
            if valid < a.input_ids.size(1):
                a.input_ids[i,valid]=eos; a.attention_mask[i,valid]=True
            elif valid > 0:
                a.input_ids[i,valid-1]=eos
    labels=torch.full_like(a.input_ids,-100); labels[a.attention_mask.bool()]=a.input_ids[a.attention_mask.bool()]
    return {"context_ids":c.input_ids,"context_mask":c.attention_mask.bool(),"question_ids":q.input_ids,"question_mask":q.attention_mask.bool(),"answer_ids":a.input_ids,"answer_mask":a.attention_mask.bool(),"labels":labels,"records":rows}
