"""Absolute positions of the QA tokens behind a memory prefix.

The training/evaluation path (:meth:`MetaLoRA.forward_qa_with_prefix`) and the generation
path (:meth:`MetaLoRA.generate_answers_with_prefix`) must number the *same* tokens the same
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
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import MetaLoRA  # noqa: E402

try:
    from transformers import Qwen3Config, Qwen3ForCausalLM
except ImportError:  # pragma: no cover - transformers is optional for the unit suite
    Qwen3Config = Qwen3ForCausalLM = None

VOCAB = 48
CONTEXT_LENGTH = 6
MEMORY_LENGTH = 2
START = CONTEXT_LENGTH + MEMORY_LENGTH


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


def _model(readout_length: int = 0) -> MetaLoRA:
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
    model = MetaLoRA(
        Qwen3ForCausalLM(config),
        rank=2,
        alpha=4.0,
        memory_length=MEMORY_LENGTH,
        decoder_hidden_size=8,
        decoder_heads=2,
        decoder_ffn_ratio=2,
        target_modules=["q_proj"],
        max_context_tokens=CONTEXT_LENGTH,
        trainable_dtype=torch.float32,
        token_recon=True,
        readout_length=readout_length,
    )
    model.eval()
    return model


def _prefix(model: MetaLoRA, contexts: int = 1):
    ids = torch.randint(0, VOCAB, (contexts, CONTEXT_LENGTH))
    mask = torch.ones(contexts, CONTEXT_LENGTH, dtype=torch.bool)
    embedding = model.qwen.get_input_embeddings()
    return model.encode_context_prefix(embedding(ids), mask)


def _record_positions(model: MetaLoRA) -> list:
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


def _train_positions(model: MetaLoRA, prefix):
    embedding = model.qwen.get_input_embeddings()
    answer_ids = torch.randint(2, VOCAB, (QUESTION_IDS.size(0), ANSWER_LENGTH))
    answer_mask = torch.ones_like(answer_ids, dtype=torch.bool)
    seen = _record_positions(model)
    model.forward_qa_with_prefix(
        prefix,
        torch.zeros(QUESTION_IDS.size(0), dtype=torch.long),
        embedding(QUESTION_IDS),
        QUESTION_MASK,
        embedding(answer_ids),
        answer_mask,
        answer_ids.clone(),
    )
    return seen


def _generation_positions(model: MetaLoRA, prefix):
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

    # The padded row numbers its real tokens 0, 1 from ``start``; the two padding slots are
    # parked at ``start`` and masked out.
    assert train[0, :4].tolist() == [START, START, START, START + 1], train[0].tolist()
    # The unpadded row keeps exactly the legacy numbering.
    assert train[1, :4].tolist() == [START, START + 1, START + 2, START + 3], train[1].tolist()
    # The answer starts right after the row's own question, not after the padded width.
    assert train[0, 4].item() == START + 2, train[0].tolist()
    assert train[1, 4].item() == START + 4, train[1].tolist()
    assert train[0, 4:].tolist() == [START + 2, START + 3, START + 4]


def test_generation_prefills_the_same_positions_as_training():
    if not _requirements_available():
        return
    model = _model()
    prefix = _prefix(model)
    train = _train_positions(model, prefix)[0]
    generation = _generation_positions(model, prefix)

    prefill = generation[0]
    assert prefill.shape == (2, 4)
    assert torch.equal(prefill, train[:, :4]), (
        f"generation prefilled {prefill.tolist()} but training supervised {train[:, :4].tolist()}"
    )


def test_the_first_generated_token_sits_where_training_supervised_it():
    """The README's original invariant: same supervised position and inference position."""
    if not _requirements_available():
        return
    model = _model()
    prefix = _prefix(model)
    train = _train_positions(model, prefix)[0]
    generation = _generation_positions(model, prefix)

    assert len(generation) >= 2, "the generation path should decode at least one token"
    first_decoded = generation[1]
    assert first_decoded.shape == (2, 1)
    assert first_decoded.flatten().tolist() == train[:, 4].tolist(), (
        f"first decoded positions {first_decoded.flatten().tolist()} != supervised "
        f"first answer positions {train[:, 4].tolist()}"
    )


def test_the_readout_and_the_answer_follow_the_real_question_length():
    if not _requirements_available():
        return
    model = _model(readout_length=8)
    prefix = _prefix(model)
    train = _train_positions(model, prefix)[0]

    readout = train[:, 4:12]
    assert readout[0].tolist() == list(range(START + 2, START + 10)), readout[0].tolist()
    assert readout[1].tolist() == list(range(START + 4, START + 12)), readout[1].tolist()
    assert train[0, 12].item() == START + 2 + 8
    assert train[1, 12].item() == START + 4 + 8
    assert train[0, 12:].tolist() == [START + 10, START + 11, START + 12]

    generation = _generation_positions(model, prefix)
    assert torch.equal(generation[0][:, :12], train[:, :12])


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
