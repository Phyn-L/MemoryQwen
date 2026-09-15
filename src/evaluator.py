from __future__ import annotations

import math
import os

import torch
from tqdm.auto import tqdm

from .losses import combine_losses, qa_loss, reconstruction_loss
from .metrics import qa_metrics


class Evaluator:
    def __init__(self, tokenizer, cfg):
        self.tokenizer, self.cfg = tokenizer, cfg

    def _batch(self, model, batch, device, start=0, end=None):
        end = end or batch["question_ids"].size(0)
        ids = {key: value.to(device) for key, value in batch.items() if key not in {"records"}}
        qa_indices = ids["qa_context_indices"][start:end]
        embed = model.qwen.get_input_embeddings()
        context_ids = ids["context_ids"].index_select(0, qa_indices)
        context_mask = ids["context_mask"].index_select(0, qa_indices)
        output = model(embed(context_ids), context_mask, embed(ids["question_ids"][start:end]), ids["question_mask"][start:end], embed(ids["answer_ids"][start:end]), ids["answer_mask"][start:end], ids["labels"][start:end])
        reconstruction = reconstruction_loss(output.reconstruction, output.context_target, output.context_mask, self.cfg.memory.reconstruction_cosine_weight, self.cfg.memory.reconstruction_loss)
        qa = qa_loss(output.logits, output.labels)
        total, _ = combine_losses(qa, reconstruction, self.cfg.memory.qa_weight, self.cfg.memory.reconstruction_weight)
        return output, qa, reconstruction, total, ids, qa_indices

    def _chunks(self, batch):
        size = self.cfg.evaluation.qa_batch_size
        return range(0, batch["question_ids"].size(0), size)

    @torch.no_grad()
    def teacher_forced(self, model, loader, device):
        model.eval(); qa_sum = reconstruction_sum = total_sum = 0.0; em = f1 = rouge = bleu = 0.0; samples = 0; weight = 0
        enabled = os.environ.get("RANK", "0") in {"0", "-1"}
        for batch in tqdm(loader, total=len(loader), desc="Validation (teacher-forced)", unit="batch", disable=not enabled):
            for start in self._chunks(batch):
                end = min(start + self.cfg.evaluation.qa_batch_size, batch["question_ids"].size(0)); output, qa, reconstruction, total, ids, _ = self._batch(model, batch, device, start, end); count = end - start
                qa_sum += float(qa) * count; reconstruction_sum += float(reconstruction) * count; total_sum += float(total) * count; weight += count
                predictions = output.logits.argmax(-1)[:, :-1]; target = output.labels[:, 1:]
                for i, record in enumerate(batch["records"][start:end]):
                    active = target[i].ne(-100); text = self.tokenizer.decode(predictions[i][active].tolist(), skip_special_tokens=True); metrics = qa_metrics(text, record.answer)
                    em += metrics["em"]; f1 += metrics["f1"]; rouge += metrics["rouge"]; bleu += metrics["bleu"]; samples += 1
        model.train(); qa_value = qa_sum / max(1, weight)
        return {"qa_loss": qa_value, "ppl": math.exp(min(qa_value, 20)), "reconstruction_loss": reconstruction_sum / max(1, weight), "loss": total_sum / max(1, weight), "em": em / max(1, samples), "f1": f1 / max(1, samples), "rouge": rouge / max(1, samples), "bleu": bleu / max(1, samples)}

    @torch.no_grad()
    def autoregressive(self, model, loader, device):
        model.eval(); sums = {"em": 0.0, "f1": 0.0, "rouge": 0.0, "bleu": 0.0}; samples = 0; enabled = os.environ.get("RANK", "0") in {"0", "-1"}
        for batch in tqdm(loader, total=len(loader), desc="Validation (autoregressive)", unit="batch", disable=not enabled):
            ids = {key: value.to(device) for key, value in batch.items() if key != "records"}; q = ids["qa_context_indices"]
            for start in self._chunks(batch):
                end = min(start + self.cfg.evaluation.qa_batch_size, q.numel()); indices = q[start:end]; context_ids = ids["context_ids"].index_select(0, indices); context_mask = ids["context_mask"].index_select(0, indices)
                generated = model.generate_answer(context_ids, ids["question_ids"][start:end], context_mask, ids["question_mask"][start:end], self.tokenizer, self.cfg.evaluation.max_new_tokens)
                for i, record in enumerate(batch["records"][start:end]):
                    metrics = qa_metrics(self.tokenizer.decode(generated[i].tolist(), skip_special_tokens=True), record.answer)
                    for key in sums: sums[key] += metrics[key]
                    samples += 1
        model.train(); result = {key: value / max(1, samples) for key, value in sums.items()}; teacher = self.teacher_forced(model, loader, device); result.update({"ppl": teacher["ppl"], "qa_loss": teacher["qa_loss"], "reconstruction_loss": teacher["reconstruction_loss"], "rouge_l": result["rouge"]}); return result
