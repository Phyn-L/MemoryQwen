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

    def _batch(self, model, batch, device):
        embed = model.qwen.get_input_embeddings()
        ids = {key: value.to(device) for key, value in batch.items() if key != "records"}
        output = model(
            embed(ids["context_ids"]), ids["context_mask"],
            embed(ids["question_ids"]), ids["question_mask"],
            embed(ids["answer_ids"]), ids["answer_mask"], ids["labels"],
        )
        qa = qa_loss(output.logits, output.labels)
        reconstruction = reconstruction_loss(
            output.reconstruction, output.context_target, output.context_mask,
            self.cfg.memory.reconstruction_cosine_weight,
            self.cfg.memory.reconstruction_loss,
        )
        total, _ = combine_losses(
            qa, reconstruction, self.cfg.memory.qa_weight,
            self.cfg.memory.reconstruction_weight,
        )
        return output, qa, reconstruction, total, ids

    @torch.no_grad()
    def teacher_forced(self, model, loader, device):
        model.eval()
        qa_sum = reconstruction_sum = total_sum = 0.0
        em = f1 = rouge = bleu_score = 0.0
        samples = batches = 0
        enabled = os.environ.get("RANK", "0") in {"0", "-1"}
        for batch in tqdm(loader, total=len(loader), desc="Validation (teacher-forced)", unit="batch", disable=not enabled):
            output, qa, reconstruction, total, _ = self._batch(model, batch, device)
            qa_sum += float(qa)
            reconstruction_sum += float(reconstruction)
            total_sum += float(total)
            batches += 1
            predictions = output.logits.argmax(-1)[:, :-1]
            target_labels = output.labels[:, 1:]
            for index, record in enumerate(batch["records"]):
                active = target_labels[index].ne(-100)
                text = self.tokenizer.decode(
                    predictions[index][active].tolist(), skip_special_tokens=True,
                )
                metrics = qa_metrics(text, record.answer)
                em += metrics["em"]
                f1 += metrics["f1"]
                rouge += metrics["rouge"]
                bleu_score += metrics["bleu"]
                samples += 1
        model.train()
        qa_value = qa_sum / max(1, batches)
        return {
            "qa_loss": qa_value,
            "ppl": math.exp(min(qa_value, 20)),
            "reconstruction_loss": reconstruction_sum / max(1, batches),
            "loss": total_sum / max(1, batches),
            "em": em / max(1, samples),
            "f1": f1 / max(1, samples),
            "rouge": rouge / max(1, samples),
            "bleu": bleu_score / max(1, samples),
        }

    @torch.no_grad()
    def autoregressive(self, model, loader, device):
        model.eval()
        sums = {"em": 0.0, "f1": 0.0, "rouge": 0.0, "bleu": 0.0}
        samples = 0
        enabled = os.environ.get("RANK", "0") in {"0", "-1"}
        for batch in tqdm(loader, total=len(loader), desc="Validation (autoregressive)", unit="batch", disable=not enabled):
            ids = {key: value.to(device) for key, value in batch.items() if key != "records"}
            generated = model.generate_answer(
                ids["context_ids"], ids["question_ids"],
                ids["context_mask"], ids["question_mask"],
                self.tokenizer, self.cfg.evaluation.max_new_tokens,
            )
            for index, record in enumerate(batch["records"]):
                metrics = qa_metrics(
                    self.tokenizer.decode(generated[index].tolist(), skip_special_tokens=True),
                    record.answer,
                )
                for key in sums:
                    sums[key] += metrics[key]
                samples += 1
        model.train()
        result = {key: value / max(1, samples) for key, value in sums.items()}
        teacher_forced = self.teacher_forced(model, loader, device)
        result.update({
            "ppl": teacher_forced["ppl"],
            "qa_loss": teacher_forced["qa_loss"],
            "reconstruction_loss": teacher_forced["reconstruction_loss"],
            "rouge_l": result["rouge"],
        })
        return result
