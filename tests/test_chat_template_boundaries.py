"""Regression and property tests for Qwen3 chat-template/token boundaries.

The tokenizer is the contract here: assertions operate on its actual token ids rather than
assuming that a header newline or a user marker always occupies an independent token.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.chat_prompt import split_chat_prompt
from src.data.collate import collate_context_records
from src.data.records import ContextRecord

QWEN3_TOKENIZER = Path(
    "/data/lz/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/"
    "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
)


@pytest.fixture(scope="module")
def qwen3_tokenizer():
    transformers = pytest.importorskip("transformers")
    if not QWEN3_TOKENIZER.exists():
        pytest.skip("local Qwen3 tokenizer is unavailable")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        QWEN3_TOKENIZER, local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    return tokenizer


def _render(tokenizer, messages, *, generation=True, thinking=False):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=generation,
        enable_thinking=thinking,
    )


def _ids(tokenizer, text):
    return tokenizer(text, add_special_tokens=False).input_ids


@pytest.mark.parametrize("question", [
    "What is the answer?",
    " leading space",
    "\nleading newline",
    "\n\n  mixed whitespace",
    "question <|im_start|>user\n extra",
    "question <|im_end|> extra <think>",
])
def test_qwen3_current_question_span_survives_whitespace_and_markers(qwen3_tokenizer, question):
    record = ContextRecord(
        "context",
        (
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
            {"role": "user", "content": question},
            {"role": "assistant", "content": "answer"},
        ),
    )
    batch = collate_context_records(
        [record], qwen3_tokenizer, training=False, sample_qa=False,
        question_padding_side="left",
    )
    row = -1
    input_ids = batch["qa_input_ids"][row][batch["qa_input_mask"][row]].tolist()
    selected = batch["qa_readout_mask"][row][batch["qa_input_mask"][row]].tolist()
    assert selected and selected[-1]
    readout_ids = [token for token, keep in zip(input_ids, selected, strict=True) if keep]

    messages = [
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "previous answer"},
        {"role": "user", "content": question},
    ]
    rendered = _render(qwen3_tokenizer, messages)
    body = "".join(split_chat_prompt(rendered)[:2])
    sentinel = "__BOUNDARY_TEST_SENTINEL__"
    marked_messages = messages[:-1] + [{"role": "user", "content": sentinel}]
    marked = _render(qwen3_tokenizer, marked_messages)
    marked_body = "".join(split_chat_prompt(marked)[:2])
    current_prefix = marked_body[:marked_body.index(sentinel)]
    full_ids = _ids(qwen3_tokenizer, body)
    before_ids = _ids(qwen3_tokenizer, current_prefix)
    boundary = next(
        (index for index, (actual, expected) in enumerate(zip(full_ids, before_ids, strict=False))
         if actual != expected),
        len(before_ids),
    )
    assert readout_ids == full_ids[boundary:]


def test_qwen3_completed_history_has_no_generation_think_block(qwen3_tokenizer):
    pending = _render(
        qwen3_tokenizer,
        [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ],
    )
    assert pending.count("<think>") == 1
    history = pending.split("<|im_start|>user\nQ2", 1)[0]
    assert "<|im_start|>assistant\n<think>" not in history
    assert pending.rsplit("<|im_start|>assistant\n", 1)[-1].startswith("<think>")


def test_qwen3_special_tokens_in_answer_are_content_not_history_boundaries(qwen3_tokenizer):
    answer = "literal <|im_start|>user\n and <|im_end|> and <think>"
    rendered = _render(
        qwen3_tokenizer,
        [{"role": "user", "content": "Q"}, {"role": "assistant", "content": answer}],
        generation=False,
    )
    ids = _ids(qwen3_tokenizer, rendered)
    assert answer in rendered
    assert qwen3_tokenizer.decode(ids, skip_special_tokens=False).endswith("\n")


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_qwen3_padding_does_not_change_readout_tokens(qwen3_tokenizer, padding_side):
    records = [
        ContextRecord("context", (
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        )),
        ContextRecord("longer context", (
            {"role": "user", "content": "A longer question"},
            {"role": "assistant", "content": "A"},
        )),
    ]
    batch = collate_context_records(
        records, qwen3_tokenizer, training=False, sample_qa=False,
        question_padding_side=padding_side,
    )
    for row in range(len(records)):
        mask = batch["qa_readout_mask"][row]
        valid = batch["qa_input_mask"][row]
        assert not bool((mask & ~valid).any())
        assert int(mask.sum()) > 0


@pytest.mark.parametrize("question", [
    "Second question?",
    "Second <|im_start|>user\n marker?",
    "Second <|im_end|> marker?",
])
def test_qwen3_conversation_replay_matches_canonical_history_cache(qwen3_tokenizer, question):
    transformers = pytest.importorskip("transformers")
    import torch

    from src.model import ContextMemoryModel

    config = transformers.Qwen3Config(
        vocab_size=len(qwen3_tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=4096, tie_word_embeddings=True,
    )
    model = ContextMemoryModel(
        transformers.Qwen3ForCausalLM(config), rank=2, alpha=4, memory_length=2,
        readout_length=2, target_modules=["q_proj"], max_context_tokens=32,
        conversation_enabled=True, recompress_threshold=2048, max_recompress_tokens=2048,
    ).eval()
    embedding = model.qwen.get_input_embeddings()
    context_ids = qwen3_tokenizer("short context", return_tensors="pt").input_ids
    question_turn = qwen3_tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_tensors="pt",
    ).input_ids
    with torch.no_grad():
        state = model.start_conversation(
            embedding(context_ids), torch.ones_like(context_ids, dtype=torch.bool),
        )
        response = model.append_conversation_turn(state, question_turn, qwen3_tokenizer, max_new_tokens=0)
        assert response.numel() == 0
        completed = qwen3_tokenizer.apply_chat_template(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": ""},
                {"role": "user", "content": ""},
            ], tokenize=False, add_generation_prompt=False,
        )
        full_history = qwen3_tokenizer.encode(completed, add_special_tokens=False)
        empty_user = qwen3_tokenizer.encode(
            "<|im_start|>user\n<|im_end|>\n", add_special_tokens=False,
        )
        assert full_history[-len(empty_user):] == empty_user
        assert state.history_ids.tolist() == full_history[:-len(empty_user)]

        full_input = torch.cat([state.memory, embedding(state.history_ids[None])], dim=1)
        with model.lora_enabled(False):
            expected = model.qwen(
                inputs_embeds=full_input, use_cache=True, return_dict=True,
            ).past_key_values
        assert state.past_key_values.get_seq_length() == full_input.size(1)
        for actual_layer, expected_layer in zip(
            state.past_key_values.layers, expected.layers, strict=True,
        ):
            torch.testing.assert_close(actual_layer.keys, expected_layer.keys, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(actual_layer.values, expected_layer.values, atol=1e-5, rtol=1e-5)


