"""The decode path: batched greedy generation, and the headless decoder body.

``generate_answers_with_prefix`` decodes many QA rows that share a context prefix as one batch.
The rewrite that removed the per-step ``torch.cat`` of the attention mask and the per-row Python
token bookkeeping must not change a single produced token, so the central test here is the
equality against the row-by-row path (``group_by_context=False``) on a padded batch.

``_transformer_body`` is the other half: the headless decoder stack the encoder and the loss-only
forwards use. It used to fall back to the full model silently, which turns an unknown
transformers wrapper into an out-of-memory error, and it must not be registered as a second
submodule (that would duplicate every backbone tensor in ``state_dict()``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.model import ContextMemoryModel  # noqa: E402

try:
    from transformers import Qwen3Config, Qwen3ForCausalLM
except ImportError:  # pragma: no cover - transformers is optional for the unit suite
    Qwen3Config = Qwen3ForCausalLM = None

VOCAB = 48
CONTEXT_LENGTH = 6
MEMORY_LENGTH = 2


class _Tokenizer:
    """Only what the generation path reads off a tokenizer."""

    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return {
            "<|im_start|>user\n": [8, 9],
            "<|im_end|>": [5],
            "<|im_start|>assistant\n": [6, 7],
        }[text]

    def decode(self, ids, skip_special_tokens=True):  # pragma: no cover - not asserted
        return "x"


def _requirements_available() -> bool:
    if Qwen3Config is None:
        print("   (skipped: transformers is not installed)")
        return False
    return True


def _model(readout_length=0) -> ContextMemoryModel:
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=VOCAB, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, intermediate_size=32, tie_word_embeddings=True,
    )
    model = ContextMemoryModel(
        Qwen3ForCausalLM(config), rank=2, alpha=4.0, memory_length=MEMORY_LENGTH,
        target_modules=["q_proj"], max_context_tokens=CONTEXT_LENGTH,
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


# The first row is a two-token question left-padded by two, so the batch exercises the mask
# path; the second row has no padding at all.
QUESTION_IDS = torch.tensor([[0, 0, 5, 6], [7, 8, 9, 10]])
QUESTION_MASK = torch.tensor([[False, False, True, True], [True, True, True, True]])
CONTEXT_INDICES = torch.zeros(QUESTION_IDS.size(0), dtype=torch.long)


def _generate(model, prefix, **kwargs):
    return model.generate_answers_with_prefix(
        prefix, CONTEXT_INDICES, QUESTION_IDS, QUESTION_MASK, _Tokenizer(), **kwargs
    )


def test_batched_generation_produces_the_same_tokens_as_row_by_row():
    if not _requirements_available():
        return
    model = _model()
    prefix = _prefix(model, contexts=1)
    batched = _generate(model, prefix, max_new_tokens=4, group_by_context=True, max_rows_per_group=2)
    row_wise = _generate(model, prefix, max_new_tokens=4, group_by_context=False)
    one_at_a_time = _generate(model, prefix, max_new_tokens=4, group_by_context=True, max_rows_per_group=1)
    assert batched.shape == row_wise.shape == one_at_a_time.shape == (2, 4)
    assert torch.equal(batched, row_wise)
    assert torch.equal(batched, one_at_a_time)


def test_generation_is_deterministic_and_respects_the_token_budget():
    if not _requirements_available():
        return
    model = _model()
    prefix = _prefix(model, contexts=1)
    first = _generate(model, prefix, max_new_tokens=3, group_by_context=True, max_rows_per_group=2)
    again = _generate(model, prefix, max_new_tokens=3, group_by_context=True, max_rows_per_group=2)
    assert torch.equal(first, again)
    assert first.shape == (2, 3)

    # A zero budget produces no tokens at all rather than a decode step.
    assert _generate(model, prefix, max_new_tokens=0).shape == (2, 0)


def test_generation_prefill_projects_only_the_last_position():
    if not _requirements_available():
        return
    model = _model()
    prefix = _prefix(model, contexts=1)
    original = model.qwen.forward
    calls = []

    def capture(*args, **kwargs):
        output = original(*args, **kwargs)
        calls.append((kwargs.get("logits_to_keep"), output.logits.size(1)))
        return output

    model.qwen.forward = capture
    try:
        _generate(model, prefix, max_new_tokens=1, group_by_context=False)
    finally:
        model.qwen.forward = original

    assert calls[0] == (1, 1)


def test_conversation_appends_turns_and_recompresses_memory():
    if not _requirements_available():
        return
    model = _model(readout_length=2)
    model.conversation_enabled = True
    model.recompress_threshold = MEMORY_LENGTH + 2 + 11
    model.max_recompress_tokens = MEMORY_LENGTH + 4
    embedding = model.qwen.get_input_embeddings()
    context = embedding(torch.randint(0, VOCAB, (1, CONTEXT_LENGTH)))
    user_prefix = embedding(torch.tensor([[8, 9]]))
    state = model.start_conversation(
        context, torch.ones(1, CONTEXT_LENGTH, dtype=torch.bool), user_prefix,
    )
    initial_cache = state.past_key_values.get_seq_length()
    assert initial_cache == user_prefix.size(1) + MEMORY_LENGTH
    turn = torch.tensor([[8, 9, 2, 3, 5, 10, 6, 7, 11, 12]])
    tokenizer = _Tokenizer()
    tokenizer.eos_token_id = None
    qwen_forward = model.qwen.forward
    seen_inputs = []

    def capture(*args, **kwargs):
        seen_inputs.append(kwargs["inputs_embeds"].detach().clone())
        return qwen_forward(*args, **kwargs)

    model.qwen.forward = capture
    resampler_forward = model.resampler.forward
    resampler_inputs = []

    def capture_resampler(hidden, mask):
        resampler_inputs.append(hidden.detach().clone())
        return resampler_forward(hidden, mask)

    model.resampler.forward = capture_resampler
    first = model.append_conversation_turn(state, turn, tokenizer, max_new_tokens=1)
    model.qwen.forward = qwen_forward
    model.resampler.forward = resampler_forward
    assert first.shape == (1,)
    assert state.history_ids.numel() == turn.numel() + first.numel()
    assert state.past_key_values.get_seq_length() > initial_cache
    assert seen_inputs[0].size(1) == 2
    assert seen_inputs[1].size(1) == 2 + 6
    assert resampler_inputs[0].size(1) == MEMORY_LENGTH + 2
    torch.testing.assert_close(resampler_inputs[0][:, :MEMORY_LENGTH], state.memory_hidden)
    before = state.memory.clone()
    second = model.append_conversation_turn(state, turn, tokenizer, max_new_tokens=1)
    assert second.shape == (1,)
    assert state.history_ids.numel() == turn.numel() + second.numel()
    assert state.past_key_values.get_seq_length() == user_prefix.size(1) + MEMORY_LENGTH + 4 + 2 + 6 + second.numel()
    assert not torch.equal(state.memory, before)


def test_start_conversation_rejects_batched_context():
    if not _requirements_available():
        return
    model = _model()
    model.conversation_enabled = True
    embedding = model.qwen.get_input_embeddings()
    context = embedding(torch.randint(0, VOCAB, (2, CONTEXT_LENGTH)))
    try:
        model.start_conversation(
            context, torch.ones(2, CONTEXT_LENGTH, dtype=torch.bool),
            embedding(torch.tensor([[8, 9]])),
        )
    except ValueError as error:
        assert "batch size 1" in str(error)
    else:
        raise AssertionError("start_conversation must reject batches larger than one")


def test_append_conversation_turn_rejects_batched_turn():
    if not _requirements_available():
        return
    model = _model()
    model.conversation_enabled = True
    embedding = model.qwen.get_input_embeddings()
    state = model.start_conversation(
        embedding(torch.randint(0, VOCAB, (1, CONTEXT_LENGTH))),
        torch.ones(1, CONTEXT_LENGTH, dtype=torch.bool),
        embedding(torch.tensor([[8, 9]])),
    )
    try:
        model.append_conversation_turn(
            state,
            torch.tensor(
                [[8, 9, 2, 3, 5, 10, 6, 7, 11, 12], [8, 9, 2, 3, 5, 10, 6, 7, 11, 12]]
            ),
            _Tokenizer(),
            1,
        )
    except ValueError as error:
        assert "batch size 1" in str(error) or "one non-empty" in str(error)
    else:
        raise AssertionError("append_conversation_turn must reject batches larger than one")


def test_append_conversation_turn_checks_generation_window():
    if not _requirements_available():
        return
    model = _model(readout_length=2)
    model.conversation_enabled = True
    embedding = model.qwen.get_input_embeddings()
    state = model.start_conversation(
        embedding(torch.randint(0, VOCAB, (1, CONTEXT_LENGTH))),
        torch.ones(1, CONTEXT_LENGTH, dtype=torch.bool),
        embedding(torch.tensor([[8, 9]])),
    )
    model.window_limit = state.past_key_values.get_seq_length() + 10
    try:
        model.append_conversation_turn(
            state, torch.tensor([[8, 9, 2, 3, 5, 10, 6, 7, 11, 12]]), _Tokenizer(), max_new_tokens=100,
        )
    except ValueError as error:
        assert "window" in str(error)
    else:
        raise AssertionError("append_conversation_turn must enforce the generation window")


def test_the_decode_loop_never_calls_a_sampler(monkeypatch):
    """Evaluation decoding is greedy: ``torch.multinomial`` (the sampler entry point) is never hit.

    A validation score must be a property of the checkpoint, not of a random draw. Both the
    prefix decoder and ``transformers``' own ``generate(do_sample=True)`` go through
    ``torch.multinomial``, so making it raise is a check that no decode path samples.
    """
    if not _requirements_available():
        return
    model = _model()
    prefix = _prefix(model, contexts=1)

    def _forbidden(*args, **kwargs):  # pragma: no cover - only runs when sampling happens
        raise AssertionError("evaluation decoding must be greedy, not sampled")

    monkeypatch.setattr(torch, "multinomial", _forbidden)
    generated = _generate(model, prefix, max_new_tokens=3, group_by_context=True)
    assert generated.shape == (2, 3)


def test_no_evaluation_decode_site_asks_for_sampling():
    """A source guard: adding a sampling flag to a decode call fails here."""
    sources = [
        REPO / "src" / "model.py",
        REPO / "scripts" / "evaluation" / "test_icl_suite.py",
        REPO / "scripts" / "test.py",
        REPO / "src" / "evaluator.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    assert "argmax" in text, "the prefix decoder picks the argmax token"
    assert "do_sample=False" in text, "the HF generate call must say so explicitly"
    for banned in ("do_sample=True", "temperature=", "top_p=", "top_k=", "num_beams="):
        assert banned not in text, f"{banned} must not appear in a decode path"


def test_rows_that_hit_eos_are_recorded_and_padded():
    if not _requirements_available():
        return
    model = _model()
    prefix = _prefix(model, contexts=1)
    tokenizer = _Tokenizer()
    original = model.qwen.forward
    calls = {"n": 0}

    def forced(*args, **kwargs):
        output = original(*args, **kwargs)
        calls["n"] += 1
        # Prefill: row 1 stops immediately. First incremental step: row 0 stops.
        row = 1 if calls["n"] == 1 else 0
        output.logits[row, -1] = -1e9
        output.logits[row, -1, tokenizer.eos_token_id] = 1e9
        return output

    model.qwen.forward = forced
    try:
        result = _generate(
            model, prefix, max_new_tokens=5, group_by_context=True, max_rows_per_group=2
        )
    finally:
        model.qwen.forward = original

    assert result.shape == (2, 2), "the width is the longest row, not the budget"
    assert result[1, 0].item() == tokenizer.eos_token_id
    assert result[1, 1].item() == tokenizer.pad_token_id
    assert result[0, 1].item() == tokenizer.eos_token_id


def test_the_decoder_body_has_no_lm_head_and_is_not_a_second_submodule():
    if not _requirements_available():
        return
    model = _model()
    body = model._transformer_body
    assert body is not model.qwen
    assert hasattr(body, "layers") and not hasattr(body, "lm_head")
    # Same object, cached, and reachable only through `qwen`: registering it as a submodule
    # would put every backbone tensor into state_dict() twice.
    assert model._transformer_body is body
    assert "_body" not in dict(model.named_modules())
    assert not any(key.startswith("_body") for key in model.state_dict())


def test_an_unknown_wrapper_is_an_error_not_a_silent_fallback():
    if not _requirements_available():
        return
    model = _model()
    original = model.qwen

    class _UnknownWrapper(torch.nn.Module):
        """A backbone whose decoder stack this code cannot locate."""

        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(4, 4)

    model.qwen = _UnknownWrapper()
    object.__setattr__(model, "_body", None)
    try:
        try:
            _ = model._transformer_body
        except RuntimeError as error:
            assert "_UnknownWrapper" in str(error) and ".layers" in str(error)
        else:
            raise AssertionError("an unknown wrapper must raise instead of falling back")
    finally:
        model.qwen = original


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
