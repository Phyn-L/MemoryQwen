"""Collate contracts: deterministic turn sampling and one tokenization per target.

Which QA turns a context contributes used to come from ``random.sample`` on the process-global
stream inside the DataLoader workers, so it depended on the worker layout: changing
``data.num_workers`` changed the training distribution, and a resumed run could not reproduce
the rows it had already consumed. It is now keyed on (record, epoch) only, which is what the
first test pins down.

The second test pins the other half of the cleanup: every supervision target is tokenized
exactly once. It used to be tokenized once for the whole batch and then again per task, which
is also how the truncation limits of the two views could drift apart.
"""
from __future__ import annotations

import pytest
import torch

from src.data import (
    ContextRecord,
    _messages_from_row,
    _trim_prompt_parts,
    _user_prefix_text,
    collate_context_records,
)


def test_trim_prompt_accepts_a_header_newline_merged_with_question():
    class _MergedHeaderTokenizer:
        def __call__(self, text, add_special_tokens=False):
            assert text == "<|im_start|>user\n"
            return type("Encoded", (), {"input_ids": [20, 21, 22]})()

    prefix, suffix = _trim_prompt_parts(
        [20, 21], list(range(30, 42)), 8, _MergedHeaderTokenizer()
    )
    assert prefix == [20, 21]
    assert suffix == [34, 35, 36, 37, 38, 39, 40, 41][-(8 - 3):]


class _Batch:
    def __init__(self, rows, masks=None):
        self.input_ids = torch.tensor(rows, dtype=torch.long)
        if masks is None:
            masks = [[1] * len(row) for row in rows]
        self.attention_mask = torch.tensor(masks, dtype=torch.long)


class _Encoded:
    """What the real tokenizer returns for one plain string: a flat id list."""

    def __init__(self, ids):
        self.input_ids = ids


class _WordTokenizer:
    """Whitespace tokenizer with the surface ``collate_context_records`` uses."""

    padding_side = "right"
    truncation_side = "right"
    eos_token_id = 1
    pad_token_id = 0
    VOCAB = {
        word: index
        for index, word in enumerate(
            ("the", "cat", "sat", "dog", "ctx", "answer", "question", "extra", "history"),
            start=2,
        )
    }
    VOCAB.update({"<|im_start|>user": 20, "<|im_start|>assistant": 21, "<|im_end|>": 22})

    def __init__(self):
        self.calls = []

    def _ids(self, text):
        return [self.VOCAB.get(word, 2) for word in str(text).split()]

    def _pad(self, rows):
        width = max((len(row) for row in rows), default=0)
        ids, masks = [], []
        for row in rows:
            pad = width - len(row)
            if self.padding_side == "left":
                ids.append([self.pad_token_id] * pad + list(row))
                masks.append([0] * pad + [1] * len(row))
            else:
                ids.append(list(row) + [self.pad_token_id] * pad)
                masks.append([1] * len(row) + [0] * pad)
        return _Batch(ids, masks)

    def __call__(self, text, add_special_tokens=False, padding=False, truncation=False,
                 max_length=None, return_tensors=None):
        self.calls.append(text)
        if isinstance(text, str):
            return _Encoded(self._truncate(self._ids(text), truncation, max_length))
        rows = [
            self._truncate(self._ids(item), truncation, max_length) for item in list(text)
        ]
        return self._pad(rows) if padding else _Batch(rows)

    def _truncate(self, ids, truncation, max_length):
        if truncation and max_length is not None and len(ids) > max_length:
            return ids[-max_length:] if self.truncation_side == "left" else ids[:max_length]
        return ids

    def pad(self, encoded, padding=True, return_tensors=None):
        return self._pad([list(row) for row in encoded["input_ids"]])

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        if sum(item["role"] == "user" for item in messages) > 1:
            return "".join(
                f"<|im_start|>{item['role']}\n{item['content']} <|im_end|>\n"
                for item in messages
            ) + "<|im_start|>assistant\n"
        return " ".join(str(item["content"]) for item in messages)


def _record(index: int, turns: int = 6) -> ContextRecord:
    conversations = [{"role": "system", "content": "ctx history"}]
    for turn in range(turns):
        conversations.append({"role": "user", "content": f"question {index} {turn}"})
        conversations.append({"role": "assistant", "content": f"answer {index} {turn}"})
    return ContextRecord(
        context=f"ctx {index}",
        conversations=tuple(conversations),
        dataset="squad",
        context_id=f"id{index}",
    )


def _collate(rows, epoch, tokenizer=None, **kwargs):
    return collate_context_records(
        rows,
        tokenizer or _WordTokenizer(),
        max_context_tokens=8,
        max_question_tokens=6,
        max_answer_tokens=4,
        max_qa_sample_per_context=2,
        training=True,
        train_stage="mix",
        epoch=epoch,
        **kwargs,
    )


def _selected(batch):
    """``{context_id: [(user, answer), ...]}`` of the QA rows of one training batch."""
    chosen = {}
    for turn in batch["records"]:
        if not turn.messages:
            continue
        chosen.setdefault(turn.context_id, []).append(
            (turn.question, turn.answer)
        )
    return chosen


def test_the_same_records_and_epoch_select_the_same_qa_turns():
    rows = [_record(index) for index in range(3)]
    assert _selected(_collate(rows, epoch=0)) == _selected(_collate(rows, epoch=0))


def test_memory_boundary_is_after_the_first_user_header():
    rendered = (
        "<|im_start|>system\nContext<|im_end|>\n"
        "<|im_start|>user\nFirst question<|im_end|>\n"
        "<|im_start|>assistant\nFirst answer<|im_end|>\n"
        "<|im_start|>user\nSecond question<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert _user_prefix_text(rendered) == (
        "<|im_start|>system\nContext<|im_end|>\n<|im_start|>user\n"
    )


def test_each_context_selects_independently_of_the_batch_it_travels_in():
    """Worker count changes the batch composition, so it must not change the selection."""
    rows = [_record(index) for index in range(4)]
    forward = _selected(_collate(rows, epoch=0))
    shuffled = _selected(_collate(list(reversed(rows)), epoch=0))
    assert forward == shuffled

    # ... and a different epoch is allowed to select differently (it is a new pass over the
    # data), while still being reproducible.
    other = _collate(rows, epoch=1)
    assert _selected(other) == _selected(_collate(rows, epoch=1))


def test_every_supervision_target_is_tokenized_exactly_once():
    rows = [_record(index) for index in range(3)]
    tokenizer = _WordTokenizer()
    batch = _collate(rows, epoch=0, tokenizer=tokenizer)
    answers = [call for call in tokenizer.calls if isinstance(call, str)]
    assert len(answers) == len(batch["records"]), "one tokenizer call per selected turn"

    # Both task views are padded from that single pass, so the batch carries the same targets
    # the QA path tokenized: the Recon target is the context, the QA target the answer(+EOS).
    qa_rows = batch["qa_target_ids"].size(0)
    recon_rows = batch["recon_target_ids"].size(0)
    assert qa_rows + recon_rows == len(batch["records"])
    assert batch["recon_labels"][:, -1].ne(-100).all()
    assert torch.equal(
        batch["recon_target_ids"][:, 0], torch.full((recon_rows,), _WordTokenizer.VOCAB["ctx"])
    )


def test_mix_emits_one_recon_row_and_qa_per_context():
    rows = [_record(index) for index in range(2)]
    batch = _collate(rows, epoch=0)
    assert batch["qa_target_ids"].size(0) == 2 * 2
    assert batch["recon_target_ids"].size(0) == 2
    assert batch["qa_context_indices"].tolist() == [0, 0, 1, 1]
    assert batch["recon_context_indices"].tolist() == [0, 1]
    assert batch["context_ids"].size(0) == 2


def test_single_turn_qa_prompt_and_answer_limits_are_applied():
    messages = _messages_from_row({
        "dataset": "squad",
        "qa_pairs": [{"question": "question extra history", "answer": "answer " * 8}],
    })
    row = ContextRecord("ctx", messages, dataset="squad", independent_qa=True)
    batch = collate_context_records(
        [row], _WordTokenizer(), max_context_tokens=8, max_question_tokens=3,
        max_answer_tokens=2, training=False, sample_qa=False,
    )
    assert batch["qa_input_mask"].sum().item() == 3
    assert batch["qa_target_mask"].sum().item() == 3  # Two answer tokens + EOS.


@pytest.mark.parametrize("training", [False, True])
def test_race_keeps_all_options_and_full_target_beyond_token_limits(training):
    answer = "answer " * 12
    messages = _messages_from_row({
        "dataset": "race",
        "qa_pairs": [{
            "question": "question",
            "answer": answer.strip(),
            "metadata": {"options": ["cat", answer.strip(), "dog", "sat"]},
        }],
    })
    race = ContextRecord("ctx", messages, dataset="race", independent_qa=True)
    tokenizer = _WordTokenizer()
    batch = collate_context_records(
        [race, _record(0, turns=1)], tokenizer,
        max_context_tokens=8, max_question_tokens=4, max_answer_tokens=4,
        training=training, train_stage="only_qa", sample_qa=False,
    )
    expected = tokenizer.apply_chat_template([
        {"role": "user", "content": messages[0].content},
    ])
    actual = batch["qa_input_ids"][0][batch["qa_input_mask"][0]].tolist()
    assert actual == tokenizer._ids(expected)
    assert batch["qa_input_mask"][1].sum() == 4
    assert batch["qa_target_mask"][0].sum() == 13  # Full option plus EOS.
    assert batch["records"][0].answer == answer.strip()
    assert tokenizer.padding_side == "right"
    assert tokenizer.truncation_side == "right"


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_current_question_span_uses_body_boundary_not_standalone_token_count(padding_side):
    class StandaloneMismatchTokenizer(_WordTokenizer):
        def _ids(self, text):
            if str(text).strip() == "question extra":
                return [99]  # Isolated tokenization has a different length from the body suffix.
            return super()._ids(text)

    tokenizer = StandaloneMismatchTokenizer()
    record = ContextRecord("ctx", (
        {"role": "user", "content": "history question"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "\n  question extra"},
        {"role": "assistant", "content": "answer"},
    ))
    batch = collate_context_records(
        [record], tokenizer, training=False, sample_qa=False,
        question_padding_side=padding_side,
    )
    actual = batch["qa_input_ids"][-1, batch["qa_readout_mask"][-1]].tolist()
    assert actual == [tokenizer.VOCAB["question"], tokenizer.VOCAB["extra"]]
    assert not (batch["qa_readout_mask"] & ~batch["qa_input_mask"]).any()


@pytest.mark.parametrize("question", [
    "question <|im_start|>user\n extra",
    "question __MEMORYQWEN_CURRENT_USER_BOUNDARY__ extra",
])
def test_current_question_span_preserves_literal_markers_in_content(question):
    tokenizer = _WordTokenizer()
    record = ContextRecord("ctx", (
        {"role": "user", "content": "history question"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": question},
        {"role": "assistant", "content": "answer"},
    ))
    batch = collate_context_records([record], tokenizer, training=False, sample_qa=False)
    assert batch["qa_input_ids"][-1, batch["qa_readout_mask"][-1]].tolist() == tokenizer._ids(question)
