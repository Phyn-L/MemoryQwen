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

    def _batch(self, model, prefix, batch, device, start=0, end=None):
        end = end or batch["question_ids"].size(0)
        ids = {
            key: value.to(device)
            for key, value in batch.items()
            if key not in {"records"}
        }
        qa_indices = ids["qa_context_indices"][start:end]
        embed = model.qwen.get_input_embeddings()
        output = model.forward_qa_with_prefix(
            prefix,
            qa_indices,
            embed(ids["question_ids"][start:end]),
            ids["question_mask"][start:end],
            embed(ids["answer_ids"][start:end]),
            ids["answer_mask"][start:end],
            ids["labels"][start:end],
        )
        qa = qa_loss(output.logits, output.labels)
        return output, qa, ids, qa_indices

    def _chunks(self, batch):
        size = self.cfg.evaluation.qa_batch_size
        return range(0, batch["question_ids"].size(0), size)

    @torch.no_grad()
    def teacher_forced(self, model, loader, device):
        model.eval()
        qa_sum = reconstruction_sum = 0.0
        em = f1 = rouge_l = bleu = 0.0
        samples = 0
        qa_weight = context_weight = 0
        enabled = os.environ.get("RANK", "0") in {"0", "-1"}
        for batch in tqdm(
            loader,
            total=len(loader),
            desc="Validation (teacher-forced)",
            unit="batch",
            disable=not enabled,
            leave=False,
        ):
            ids = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "records"
            }
            embed = model.qwen.get_input_embeddings()
            prefix = model.encode_context_prefix(
                embed(ids["context_ids"]), ids["context_mask"]
            )
            reconstruction = reconstruction_loss(
                prefix.reconstruction,
                prefix.context_target,
                prefix.context_mask,
                self.cfg.memory.reconstruction_cosine_weight,
                self.cfg.memory.reconstruction_loss,
            )
            contexts = ids["context_ids"].size(0)
            reconstruction_sum += float(reconstruction) * contexts
            context_weight += contexts
            for start in self._chunks(batch):
                end = min(
                    start + self.cfg.evaluation.qa_batch_size,
                    batch["question_ids"].size(0),
                )
                output, qa, ids, _ = self._batch(
                    model, prefix, batch, device, start, end
                )
                count = end - start
                qa_sum += float(qa) * count
                qa_weight += count
                predictions = output.logits.argmax(-1)[:, :-1]
                target = output.labels[:, 1:]
                for i, record in enumerate(batch["records"][start:end]):
                    active = target[i].ne(-100)
                    text = self.tokenizer.decode(
                        predictions[i][active].tolist(), skip_special_tokens=True
                    )
                    metrics = qa_metrics(text, record.answer)
                    em += metrics["em"]
                    f1 += metrics["f1"]
                    rouge_l += metrics["rouge_l"]
                    bleu += metrics["bleu"]
                    samples += 1
        model.train()
        qa_value = qa_sum / max(1, qa_weight)
        reconstruction_value = reconstruction_sum / max(1, context_weight)
        total_value = (
            self.cfg.memory.qa_weight * qa_value
            + self.cfg.memory.reconstruction_weight * reconstruction_value
        )
        return {
            "qa_loss": qa_value,
            "ppl": math.exp(min(qa_value, 20)),
            "reconstruction_loss": reconstruction_value,
            "loss": total_value,
            "em": em / max(1, samples),
            "f1": f1 / max(1, samples),
            "rouge_l": rouge_l / max(1, samples),
            "bleu": bleu / max(1, samples),
        }

    @torch.no_grad()
    def autoregressive(self, model, loader, device):
        model.eval()
        sums = {"em": 0.0, "f1": 0.0, "rouge_l": 0.0, "bleu": 0.0}
        samples = 0
        enabled = os.environ.get("RANK", "0") in {"0", "-1"}
        for batch in tqdm(
            loader,
            total=len(loader),
            desc="Validation (autoregressive)",
            unit="batch",
            disable=not enabled,
            leave=False,
        ):
            ids = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "records"
            }
            q = ids["qa_context_indices"]
            embed = model.qwen.get_input_embeddings()
            prefix = model.encode_context_prefix(
                embed(ids["context_ids"]), ids["context_mask"]
            )
            for start in self._chunks(batch):
                end = min(start + self.cfg.evaluation.qa_batch_size, q.numel())
                indices = q[start:end]
                generated = model.generate_answer_with_prefix(
                    prefix,
                    indices,
                    ids["question_ids"][start:end],
                    ids["question_mask"][start:end],
                    self.tokenizer,
                    self.cfg.evaluation.max_new_tokens,
                )
                for i, record in enumerate(batch["records"][start:end]):
                    metrics = qa_metrics(
                        self.tokenizer.decode(
                            generated[i].tolist(), skip_special_tokens=True
                        ),
                        record.answer,
                    )
                    for key in sums:
                        sums[key] += metrics[key]
                    samples += 1
        model.train()
        result = {key: value / max(1, samples) for key, value in sums.items()}
        teacher = self.teacher_forced(model, loader, device)
        result.update(
            {
                "ppl": teacher["ppl"],
                "qa_loss": teacher["qa_loss"],
                "reconstruction_loss": teacher["reconstruction_loss"],
            }
        )
        return result
