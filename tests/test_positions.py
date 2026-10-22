"""Absolute positions of task, QA, readout, and answer tokens behind memory.

The training/evaluation path (:meth:`ContextMemoryModel.forward_with_memory_prefix`) and the generation
path (:meth:`ContextMemoryModel.generate_answers_with_prefix`) must number the *same* tokens the same
way, otherwise the memory-to-question distance the model sees while learning is not the one
it sees while answering -- and RoPE attention depends on exactly that distance.

The old training path used one ``arange`` over the padded question block, so a row with
``pad`` left-padding tokens put its real question at ``start + pad + i`` and its answer at
``start + padded_question + readout``, while generation compacts the padding away and uses
``start + i`` / ``start + valid + readout``. A micro-model showed the first-answer logits
differing by ~9e-2 with padding and ~3e-8 without it. These tests pin the two paths together
per row, and pin the "no padding => unchanged" case so the historical single-row behaviour is
not lost.

Run with ``python tests/test_positions.py`` or ``pytest tests``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import ContextMemoryModel  # noqa: E402

try:
    from transformers import Qwen3Config, Qwen3ForCausalLM
except ImportError:  # pragma: no cover - transformers is optional for the unit suite
    Qwen3Config = Qwen3ForCausalLM = None

VOCAB = 48
CONTEXT_LENGTH = 6
MEMORY_LENGTH = 2
START = MEMORY_LENGTH


class _Tokenizer:
    """Only what the generation path reads off a tokenizer."""

    pad_token_id = 0
    eos_token_id = 1

    def decode(self, ids, skip_special_tokens=True):  # pragma: no cover - not asserted
        return "x"


def _requirements_available() -> bool:
    if Qwen3Config is None:
        print("   (skipped: transformers is not installed)")
        return False
    return True


def _model(readout_length: int = 0) -> ContextMemoryModel:
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=32,
        tie_word_embeddings=True,
    )
    model = ContextMemoryModel(
        Qwen3ForCausalLM(config),
        rank=2,
        alpha=4.0,
        memory_length=MEMORY_LENGTH,
                                target_modules=["q_proj"],
        max_context_tokens=CONTEXT_LENGTH,
        trainable_dtype=torch.float32,
        recon_user_token_ids=(2,), recon_answer_prefix_token_ids=(3, 4),
                readout_length=readout_length,
    )
    model.eval()
    return model


def _prefix(model: ContextMemoryModel, contexts: int = 1):
    ids = torch.randint(0, VOCAB, (contexts, CONTEXT_LENGTH))
    mask = torch.ones(contexts, CONTEXT_LENGTH, dtype=torch.bool)
    embedding = model.qwen.get_input_embeddings()
    return model.encode_context_prefix(embedding(ids), mask)


def _record_positions(model: ContextMemoryModel) -> list:
    """Positions the backbone is called with, in call order (prefill first)."""
    seen: list = []
    original = model.qwen.forward

    def spy(*args, **kwargs):
        positions = kwargs.get("position_ids")
        seen.append(None if positions is None else positions.detach().clone())
        output = original(*args, **kwargs)
        # Keep greedy decoding away from EOS: the generation path stops right after it
        # emits EOS, and this test needs to observe the incremental step's positions.
        output.logits[..., _Tokenizer.eos_token_id] = -1e9
        return output

    model.qwen.forward = spy
    return seen


# Rows: [0, 0, 5, 6] is a two-token question left-padded by two; the second row has no
# padding at all, so it doubles as the "legacy numbering" control.
QUESTION_IDS = torch.tensor([[0, 0, 5, 6], [7, 8, 9, 10]])
QUESTION_MASK = torch.tensor([[False, False, True, True], [True, True, True, True]])
ANSWER_LENGTH = 3


def _train_positions(model: ContextMemoryModel, prefix):
    embedding = model.qwen.get_input_embeddings()
    answer_ids = torch.randint(2, VOCAB, (QUESTION_IDS.size(0), ANSWER_LENGTH))
    answer_mask = torch.ones_like(answer_ids, dtype=torch.bool)
    seen = _record_positions(model)
    model.forward_with_memory_prefix(
        prefix,
        torch.zeros(QUESTION_IDS.size(0), dtype=torch.long),
        embedding(QUESTION_IDS),
        QUESTION_MASK,
        embedding(answer_ids),
        answer_mask,
        answer_ids.clone(),
    )
    return seen


def _generation_positions(model: ContextMemoryModel, prefix):
    seen = _record_positions(model)
    model.generate_answers_with_prefix(
        prefix,
        torch.zeros(QUESTION_IDS.size(0), dtype=torch.long),
        QUESTION_IDS,
        QUESTION_MASK,
        _Tokenizer(),
        max_new_tokens=2,
        group_by_context=True,
        max_rows_per_group=QUESTION_IDS.size(0),
    )
    return seen


def test_training_ignores_left_padding_when_numbering_the_question():
    if not _requirements_available():
        return
    model = _model()
    prefix = _prefix(model)
    train = _train_positions(model, prefix)[0]

    # Full sequence is [memory, question, answer]; masked question padding has position 0.
    assert train[0, 2:6].tolist() == [0, 0, 2, 3], train[0].tolist()
    assert train[1, 2:6].tolist() == [2, 3, 4, 5], train[1].tolist()
    assert train[0, 6:].tolist() == [4, 5, 6], train[0].tolist()
    assert train[1, 6:].tolist() == [6, 7, 8], train[1].tolist()


def test_the_qa_block_is_numbered_by_the_shared_helper():
    """Training numbering is ``compact_positions`` of the concatenated block mask.

    The helper is the only place that decides where a token sits, so the QA block, the read-out
    block and the answer block cannot drift apart from generation or from context encoding.
    """
    if not _requirements_available():
        return
    from src.model import compact_positions

    model = _model(readout_length=1)
    prefix = _prefix(model, contexts=QUESTION_IDS.size(0))
    blocks = torch.cat([
        QUESTION_MASK.new_ones(QUESTION_IDS.size(0), MEMORY_LENGTH),
        QUESTION_MASK,
        QUESTION_MASK.new_ones(QUESTION_IDS.size(0), 1 + ANSWER_LENGTH),
    ], dim=1)
    expected = compact_positions(blocks, 0)
    got = _train_positions(model, prefix)[0]
    torch.testing.assert_close(got, expected)


def test_context_encoding_is_independent_of_the_padding_side():
    """The same content tokenized with left vs right padding must yield the same memory.

    Context positions used to be a plain ``arange``, which is only correct for right padding,
    and Qwen3's tokenizer config does not pin ``padding_side``: switching to a tokenizer whose
    default is left would have silently shifted every context position.
    """
    if not _requirements_available():
        return
    model = _model()
    embedding = model.qwen.get_input_embeddings()
    left_ids = torch.tensor([[0, 0, 3, 4, 5, 6]])
    left_mask = torch.tensor([[False, False, True, True, True, True]])
    right_ids = torch.tensor([[3, 4, 5, 6, 0, 0]])
    right_mask = torch.tensor([[True, True, True, True, False, False]])

    left = model.encode_context_prefix(embedding(left_ids), left_mask)
    right = model.encode_context_prefix(embedding(right_ids), right_mask)
    torch.testing.assert_close(left.memory, right.memory)


def test_generation_prefills_the_same_positions_as_training():
    if not _requirements_available():
        return
    model = _model()
    prefix = _prefix(model)
    train = _train_positions(model, prefix)[0]
    generation = _generation_positions(model, prefix)

    prefill = generation[0]
    assert prefill.shape == (2, 6)
    assert torch.equal(prefill, train[:, :6]), (
        f"generation prefilled {prefill.tolist()} but training supervised {train[:, :6].tolist()}"
    )


def test_the_first_generated_token_sits_where_training_supervised_it():
    """Training and generation use the same position for the first answer token."""
    if not _requirements_available():
        return
    model = _model()
    prefix = _prefix(model)
    train = _train_positions(model, prefix)[0]
    generation = _generation_positions(model, prefix)

    assert len(generation) >= 2, "the generation path should decode at least one token"
    first_decoded = generation[1]
    assert first_decoded.shape == (2, 1)
    assert first_decoded.flatten().tolist() == train[:, 6].tolist(), (
        f"first decoded positions {first_decoded.flatten().tolist()} != supervised "
        f"first answer positions {train[:, 6].tolist()}"
    )


def test_readout_and_assistant_header_match_between_training_and_generation():
    if not _requirements_available():
        return
    model = _model(readout_length=2)
    prefix = _prefix(model)
    embedding = model.qwen.get_input_embeddings()
    user_ids = torch.tensor([[11, 12], [11, 12]])
    header_ids = torch.tensor([[13, 14, 15], [13, 14, 15]])
    answer_ids = torch.randint(2, VOCAB, (2, ANSWER_LENGTH))
    captures = []
    original = model.qwen.forward

    def capture(*args, **kwargs):
        captures.append((kwargs["inputs_embeds"].detach().clone(), kwargs["position_ids"].detach().clone()))
        output = original(*args, **kwargs)
        output.logits[..., _Tokenizer.eos_token_id] = -1e9
        return output

    model.qwen.forward = capture
    model.forward_with_memory_prefix(
        prefix, torch.zeros(2, dtype=torch.long), embedding(QUESTION_IDS), QUESTION_MASK,
        embedding(answer_ids), torch.ones_like(answer_ids, dtype=torch.bool), answer_ids,
        prompt_prefix_embeds=embedding(user_ids),
        prompt_prefix_mask=torch.ones_like(user_ids, dtype=torch.bool),
        answer_prefix_embeds=embedding(header_ids),
        answer_prefix_mask=torch.ones_like(header_ids, dtype=torch.bool),
    )
    train_embeds, train_positions = captures.pop()
    model.generate_answers_with_prefix(
        prefix, torch.zeros(2, dtype=torch.long), QUESTION_IDS, QUESTION_MASK,
        _Tokenizer(), max_new_tokens=1,
        question_prefix_ids=user_ids,
        question_prefix_mask=torch.ones_like(user_ids, dtype=torch.bool),
        answer_prefix_ids=header_ids,
        answer_prefix_mask=torch.ones_like(header_ids, dtype=torch.bool),
    )
    generation_embeds, generation_positions = captures[0]
    prefill_length = generation_embeds.size(1)
    torch.testing.assert_close(generation_embeds, train_embeds[:, :prefill_length])
    torch.testing.assert_close(generation_positions, train_positions[:, :prefill_length])


def test_the_readout_and_the_answer_follow_the_real_question_length():
    if not _requirements_available():
        return
    model = _model(readout_length=8)
    prefix = _prefix(model)
    train = _train_positions(model, prefix)[0]

    readout = train[:, 6:14]
    assert readout[0].tolist() == list(range(4, 12)), readout[0].tolist()
    assert readout[1].tolist() == list(range(6, 14)), readout[1].tolist()
    assert train[0, 14:].tolist() == [12, 13, 14]
    assert train[1, 14:].tolist() == [14, 15, 16]

    generation = _generation_positions(model, prefix)
    assert torch.equal(generation[0][:, :14], train[:, :14])


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
            except Exception as error:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(error).__name__}: {error}")
            else:
                print(f"ok   {name}")
    print("\nVERDICT:", "ALL PASSED" if not failures else f"{failures} FAILED")
    raise SystemExit(1 if failures else 0)
