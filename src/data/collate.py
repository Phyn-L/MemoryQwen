"""Tokenize structured records and assemble training/evaluation batches."""
from __future__ import annotations

import hashlib
import random

import torch

from .chat_prompt import _user_prefix_text, split_chat_prompt
from .records import ConversationTurn


def _encode(tokenizer, texts, max_length, padding_side=None):
    previous = tokenizer.padding_side
    if padding_side is not None:
        tokenizer.padding_side = padding_side
    try:
        return tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
    finally:
        tokenizer.padding_side = previous

def _trim_prompt_parts(prefix_ids, suffix_ids, max_length, tokenizer):
    """Cap a single-turn prompt while retaining the user header and latest question text."""
    if len(prefix_ids) + len(suffix_ids) <= max_length:
        return prefix_ids, suffix_ids
    header_ids = tokenizer("<|im_start|>user\n", add_special_tokens=False).input_ids
    header_ids = list(header_ids)
    if not prefix_ids:
        return [], suffix_ids[-max_length:]
    # The final header newline may merge with the first question characters, so the
    # independently tokenized prefix can end at a proper prefix of ``header_ids``.
    # Validate only the part that is present and reserve the full logical header span.
    matched = max(
        (size for size in range(1, min(len(prefix_ids), len(header_ids)) + 1)
         if prefix_ids[-size:] == header_ids[:size]),
        default=0,
    )
    logical_header_length = len(header_ids) if matched else 0
    suffix_budget = max(0, max_length - logical_header_length)
    suffix_ids = suffix_ids[-suffix_budget:] if suffix_budget else []
    prefix_budget = max_length - len(suffix_ids)
    return prefix_ids[-prefix_budget:], suffix_ids

def _turn_rng(record, epoch):
    """Deterministic RNG for the QA turns of one context in one epoch.

    Keyed on the record (dataset + id + context text) and the epoch, so which QA turns a
    context contributes no longer depends on which DataLoader worker happened to pick it up,
    nor on how many workers there are. ``random.sample`` on the process-global stream did
    depend on both: changing ``data.num_workers`` silently changed the training distribution,
    and a resumed run could not reproduce the rows it had already consumed.
    """
    key = "\x00".join(
        (str(record.dataset), str(record.context_id), record.context, str(int(epoch)))
    ).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.blake2s(key, digest_size=8).digest(), "big"))

def _sample_turns(record, turns, count, epoch):
    """Pick ``count`` turns of one context, deterministically (a permutation when it is all)."""
    count = min(int(count), len(turns))
    if count <= 0 or not turns:
        return []
    return _turn_rng(record, epoch).sample(turns, count)

def collate_context_records(
    rows,
    tokenizer,
    max_context_tokens=2048,
    max_question_tokens=96,
    max_answer_tokens=64,
    append_eos=True,
    chat_template_enable_thinking=False,
    max_qa_sample_per_context=4,
    sample_qa=True,
    question_padding_side="left",
    training=False,
    train_stage="mix",
    epoch=0,
):
    """Build memory-prefix QA batches from structured conversations.

    Every selected turn is rendered as ``history + current user + generation prompt``.
    The current assistant content is kept as the target, so PWC history is visible while
    the answer being supervised is never leaked in the input.

    ``epoch`` only feeds the deterministic QA-turn sampling (:func:`_turn_rng`); the same
    records and epoch always produce the same batch, whatever the worker count.
    """
    if not getattr(tokenizer, "apply_chat_template", None):
        raise ValueError("conversation training requires tokenizer.apply_chat_template")
    if train_stage not in {"two_stage", "only_qa", "only_recon", "mix"}:
        raise ValueError("train_stage must be two_stage, only_qa, only_recon or mix")
    contexts = [row.context for row in rows]
    c = _encode(tokenizer, contexts, max_context_tokens)
    selected: list[ConversationTurn] = []
    mapping: list[int] = []
    task_ids: list[int] = []
    for row_index, record in enumerate(rows):
        turns = list(record.turns())
        if training and train_stage in {"two_stage", "only_qa", "mix"} and not turns:
            raise ValueError(
                "conversation training requires at least one assistant turn"
            )
        if training and train_stage in {"two_stage", "mix"}:
            qa_turns = _sample_turns(record, turns, max_qa_sample_per_context, epoch)
            chosen = [
                *qa_turns,
                ConversationTurn(
                    (),
                    record.context,
                    dataset=record.dataset,
                    context_id=record.context_id,
                ),
            ]
            chosen_tasks = [1] * len(qa_turns) + [0]  # 1=QA, 0=Recon
        elif training and train_stage == "only_recon":
            chosen = [
                ConversationTurn(
                    (),
                    record.context,
                    dataset=record.dataset,
                    context_id=record.context_id,
                )
            ]
            chosen_tasks = [0]
        elif training:
            chosen = _sample_turns(record, turns, max_qa_sample_per_context, epoch)
            chosen_tasks = [1] * len(chosen)
        else:
            chosen = (
                _sample_turns(record, turns, max_qa_sample_per_context, epoch)
                if sample_qa else turns
            )
            chosen_tasks = [1] * len(chosen)
        selected.extend(chosen)
        mapping.extend([row_index] * len(chosen))
        task_ids.extend(chosen_tasks)
    if not selected:
        raise ValueError("conversation batch contains no assistant turns")

    rendered = []
    for turn, task in zip(selected, task_ids, strict=True):
        if task == 0:
            rendered.append("")
            continue
        messages = turn.messages
        payload = [{"role": item.role, "content": item.content} for item in messages]
        text = tokenizer.apply_chat_template(
            payload,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=chat_template_enable_thinking,
        )
        rendered.append(text)
    # The template already contains role markers and the assistant generation marker.
    qa_rendered = [text for text, task in zip(rendered, task_ids, strict=True) if task == 1]
    qa_turns = [turn for turn, task in zip(selected, task_ids, strict=True) if task == 1]
    prefix_rows, suffix_rows, header_rows, readout_starts = [], [], [], []
    if qa_rendered:
        current_prefix_texts = []
        for turn, text in zip(qa_turns, qa_rendered, strict=True):
            sentinel = "__MEMORYQWEN_CURRENT_USER_BOUNDARY__"
            while sentinel in text:
                sentinel += "_"
            payload = [{"role": item.role, "content": item.content} for item in turn.messages]
            current_user = max(i for i, item in enumerate(payload) if item["role"] == "user")
            payload[current_user]["content"] = sentinel
            marked = tokenizer.apply_chat_template(
                payload, tokenize=False, add_generation_prompt=True,
                enable_thinking=chat_template_enable_thinking,
            )
            current_prefix_texts.append(marked[:marked.index(sentinel)])
        prefix_texts = [_user_prefix_text(text) for text in qa_rendered]
        body_texts = ["".join(split_chat_prompt(text)[:2]) for text in qa_rendered]
        all_encoded = tokenizer(qa_rendered, add_special_tokens=False, padding=True)
        prefix_encoded = tokenizer(prefix_texts, add_special_tokens=False, padding=True)
        body_encoded = tokenizer(body_texts, add_special_tokens=False, padding=True)
        current_encoded = tokenizer(current_prefix_texts, add_special_tokens=False, padding=True)
        current_prefixes = [
            [value for value, keep in zip(ids, mask, strict=True) if keep]
            for ids, mask in zip(current_encoded.input_ids, current_encoded.attention_mask, strict=True)
        ]
        all_ids = [
            [value for value, keep in zip(ids, mask, strict=True) if keep]
            for ids, mask in zip(all_encoded.input_ids, all_encoded.attention_mask, strict=True)
        ]
        encoded_prefixes = [
            [value for value, keep in zip(ids, mask, strict=True) if keep]
            for ids, mask in zip(
                prefix_encoded.input_ids, prefix_encoded.attention_mask, strict=True
            )
        ]
        encoded_bodies = [
            [value for value, keep in zip(ids, mask, strict=True) if keep]
            for ids, mask in zip(body_encoded.input_ids, body_encoded.attention_mask, strict=True)
        ]
    else:
        all_ids, encoded_prefixes, encoded_bodies, current_prefixes = [], [], [], []
    for row_index, (ids, prefix, body, current_prefix) in enumerate(
        zip(all_ids, encoded_prefixes, encoded_bodies, current_prefixes, strict=True)
    ):
        ids, prefix = list(ids), list(prefix)
        # A header-final newline can merge with leading question whitespace. Keep the
        # shared token in the question body instead of losing part of the question.
        prefix_length = next(
            (i for i, (actual, expected) in enumerate(zip(ids, prefix, strict=False)) if actual != expected),
            len(prefix),
        )
        prefix = ids[:prefix_length]
        if ids[:len(body)] != body:
            raise ValueError("chat tokenizer does not preserve the user-end token boundary")
        suffix = ids[len(prefix):len(body)]
        current_start = next(
            (i for i, (actual, expected) in enumerate(zip(ids, current_prefix, strict=False))
             if actual != expected),
            len(current_prefix),
        )
        turn = qa_turns[row_index]
        single_turn_qa = len(turn.messages) <= 2 and turn.dataset != "race"
        if single_turn_qa:
            prefix, suffix = _trim_prompt_parts(
                prefix, suffix, max_question_tokens, tokenizer
            )
        prefix_rows.append(prefix)
        suffix_rows.append(suffix)
        header_rows.append(ids[len(body):])
        readout_starts.append(0 if single_turn_qa else max(0, current_start - len(prefix)))
    if qa_rendered:
        previous_padding = tokenizer.padding_side
        tokenizer.padding_side = question_padding_side
        try:
            prefix_encoded = tokenizer.pad(
                {"input_ids": prefix_rows}, padding=True, return_tensors="pt",
            )
            suffix_encoded = tokenizer.pad(
                {"input_ids": suffix_rows}, padding=True, return_tensors="pt",
            )
            header_encoded = tokenizer.pad(
                {"input_ids": header_rows}, padding=True, return_tensors="pt",
            )
        finally:
            tokenizer.padding_side = previous_padding
        q_prefix = torch.zeros(
            (len(selected), prefix_encoded.input_ids.size(1)),
            dtype=prefix_encoded.input_ids.dtype,
        )
        q_prefix_mask = torch.zeros_like(q_prefix, dtype=torch.bool)
        q = torch.zeros(
            (len(selected), suffix_encoded.input_ids.size(1)),
            dtype=suffix_encoded.input_ids.dtype,
        )
        q_mask = torch.zeros_like(q, dtype=torch.bool)
        q_header = torch.zeros(
            (len(selected), header_encoded.input_ids.size(1)), dtype=q.dtype,
        )
        q_header_mask = torch.zeros_like(q_header, dtype=torch.bool)
        qa_rows = torch.tensor([task == 1 for task in task_ids])
        q_prefix[qa_rows] = prefix_encoded.input_ids
        q_prefix_mask[qa_rows] = prefix_encoded.attention_mask.bool()
        q[qa_rows] = suffix_encoded.input_ids
        q_mask[qa_rows] = suffix_encoded.attention_mask.bool()
        q_header[qa_rows] = header_encoded.input_ids
        q_header_mask[qa_rows] = header_encoded.attention_mask.bool()
    else:
        q_prefix = torch.empty((len(selected), 0), dtype=c.input_ids.dtype)
        q_prefix_mask = torch.empty((len(selected), 0), dtype=torch.bool)
        q = torch.empty((len(selected), 0), dtype=c.input_ids.dtype)
        q_mask = torch.empty((len(selected), 0), dtype=torch.bool)
        q_header = torch.empty((len(selected), 0), dtype=c.input_ids.dtype)
        q_header_mask = torch.empty((len(selected), 0), dtype=torch.bool)

    # Every supervision target is tokenized exactly once, here. Both task views below only
    # re-pad subsets of this list -- re-tokenizing per task was pure duplicated work and a
    # second place for the truncation limits to drift.
    target_rows = []
    for turn, task in zip(selected, task_ids, strict=True):
        cap_qa_answer = task == 1 and len(turn.messages) <= 2 and turn.dataset != "race"
        target_rows.append(tokenizer(
            turn.answer,
            truncation=task == 0 or cap_qa_answer,
            max_length=max_context_tokens if task == 0 else max_answer_tokens,
            add_special_tokens=False,
        ).input_ids)

    task_tensor = torch.tensor(task_ids, dtype=torch.long)

    def pack(task_id: int):
        """One task's rows: right-padded targets, EOS appended, padding masked out of the labels.

        The padding + EOS + label rule lives here once and serves both schemas: the training
        batch (QA and Recon rows) and the evaluation batch (QA rows only). ``append_eos``
        applies to every supervision row, QA and Recon alike.
        """
        indices = torch.nonzero(task_tensor.eq(task_id), as_tuple=False).flatten()
        if indices.numel() == 0:
            return {
                "input_ids": q.new_empty((0, q.size(1) if task_id == 1 else 0)),
                "input_mask": q_mask.new_empty((0, q_mask.size(1) if task_id == 1 else 0)),
                "prefix_ids": q_prefix.new_empty((0, q_prefix.size(1) if task_id == 1 else 0)),
                "prefix_mask": q_prefix_mask.new_empty((0, q_prefix_mask.size(1) if task_id == 1 else 0)),
                "answer_prefix_ids": q_header.new_empty((0, q_header.size(1))),
                "answer_prefix_mask": q_header_mask.new_empty((0, q_header_mask.size(1))),
                "target_ids": q.new_empty((0, 0)),
                "target_mask": q_mask.new_empty((0, 0)),
                "labels": q.new_empty((0, 0)),
                "context_indices": indices,
            }
        token_lists = [target_rows[index] for index in indices.tolist()]
        previous = tokenizer.padding_side
        tokenizer.padding_side = "right"
        try:
            target = tokenizer.pad(
                {"input_ids": token_lists}, padding=True, return_tensors="pt"
            )
        finally:
            tokenizer.padding_side = previous
        if append_eos and getattr(tokenizer, "eos_token_id", None) is not None:
            eos = int(tokenizer.eos_token_id)
            ids = torch.full(
                (target.input_ids.size(0), target.input_ids.size(1) + 1),
                int(tokenizer.pad_token_id),
                dtype=target.input_ids.dtype,
            )
            mask = torch.zeros_like(ids)
            for row in range(target.input_ids.size(0)):
                valid = int(target.attention_mask[row].sum())
                ids[row, :valid] = target.input_ids[row, :valid]
                mask[row, :valid + 1] = 1
                ids[row, valid] = eos
            target.input_ids, target.attention_mask = ids, mask
        target_labels = torch.full_like(target.input_ids, -100)
        target_labels[target.attention_mask.bool()] = target.input_ids[
            target.attention_mask.bool()
        ]
        return {
            "input_ids": q[indices] if task_id == 1 else q.new_empty((indices.numel(), 0)),
            "input_mask": (
                q_mask[indices] if task_id == 1 else q_mask.new_empty((indices.numel(), 0))
            ),
            "prefix_ids": (
                q_prefix[indices] if task_id == 1 else q_prefix.new_empty((indices.numel(), 0))
            ),
            "prefix_mask": (
                q_prefix_mask[indices]
                if task_id == 1 else q_prefix_mask.new_empty((indices.numel(), 0))
            ),
            "target_ids": target.input_ids,
            "answer_prefix_ids": q_header[indices],
            "answer_prefix_mask": q_header_mask[indices],
            "target_mask": target.attention_mask.bool(),
            "labels": target_labels,
            "context_indices": torch.tensor(
                [mapping[index] for index in indices.tolist()], dtype=torch.long
            ),
        }

    qa = pack(1)
    # The answer reader keeps history; the question resampler sees only the latest user body.
    readout_mask = torch.zeros_like(qa["input_mask"])
    for index, start in enumerate(readout_starts):
        valid_indices = qa["input_mask"][index].nonzero(as_tuple=True)[0]
        readout_mask[index, valid_indices[start:]] = True
    # One batch schema for training and evaluation: the evaluator and ``scripts/test.py``
    # read the same names, and the Recon half is simply absent from an evaluation batch.
    batch = {
        "context_ids": c.input_ids,
        "context_mask": c.attention_mask.bool(),
        "qa_input_ids": qa["input_ids"],
        "qa_input_mask": qa["input_mask"],
        "qa_readout_mask": readout_mask,
        "qa_prefix_ids": qa["prefix_ids"],
        "qa_prefix_mask": qa["prefix_mask"],
        "qa_answer_prefix_ids": qa["answer_prefix_ids"],
        "qa_answer_prefix_mask": qa["answer_prefix_mask"],
        "qa_target_ids": qa["target_ids"],
        "qa_target_mask": qa["target_mask"],
        "qa_labels": qa["labels"],
        "qa_context_indices": qa["context_indices"],
        "records": selected,
    }
    if training:
        recon = pack(0)
        batch.update(
            {
                "recon_input_ids": recon["input_ids"],
                "recon_input_mask": recon["input_mask"],
                "recon_target_ids": recon["target_ids"],
                "recon_target_mask": recon["target_mask"],
                "recon_labels": recon["labels"],
                "recon_context_indices": recon["context_indices"],
            }
        )
    return batch

collate_fn = collate_context_records
