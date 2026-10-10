"""Verify prompt placement in the actual encoder/decoder token tensors."""
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from src.data import ContextRecord, QARecord, collate_context_records
from src.icl_baseline import system_prompt_for


@pytest.fixture
def tokenizer():
    words = "[PAD] [UNK] evidence question answer <system> <user> <assistant>".split()
    words += list(dict.fromkeys(" ".join(system_prompt_for(d, v) for d, v in [
        ("squad", None), ("hotpotqa", None), ("ms_marco", "ms_marco_v2_1")
    ]).split()))
    backend = Tokenizer(models.WordLevel(dict(zip(dict.fromkeys(words), range(len(set(words))))), unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]")
    tok.chat_template = "{% for message in messages %}<{{ message['role'] }}> {{ message['content'] }} {% endfor %}<assistant>"
    return tok


@pytest.mark.parametrize("dataset,version", [("squad", None), ("hotpotqa", None), ("ms_marco", "ms_marco_v2_1")])
def test_prompt_placement_preserves_evidence_and_qa_mapping(tokenizer, dataset, version):
    rows = [ContextRecord("evidence evidence", (QARecord("question", "answer"), QARecord("question", "answer")), dataset)]
    kwargs = dict(max_context_tokens=2, max_question_tokens=1, append_eos=False, sample_qa=False, source_version=version)
    plain = collate_context_records(rows, tokenizer, **kwargs)
    context = collate_context_records(rows, tokenizer, system_prompt_mode="context", **kwargs)
    separate = collate_context_records(rows, tokenizer, system_prompt_mode="separate", **kwargs)
    expected = tokenizer(system_prompt_for(dataset, version) + "\n\n", add_special_tokens=False).input_ids
    assert context["context_ids"][0].tolist() == expected + plain["context_ids"][0].tolist()
    assert context["input_ids"].equal(plain["input_ids"])
    assert separate["context_ids"].equal(plain["context_ids"])
    message = tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt_for(dataset, version)},
        {"role": "user", "content": "question"},
    ], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert separate["input_ids"][0].tolist() == tokenizer(message, add_special_tokens=False).input_ids
    for batch in (context, separate):
        assert batch["qa_context_indices"].tolist() == [0, 0]
        assert batch["labels"].equal(plain["labels"])
        assert batch["context_mask"].all() and batch["input_mask"].all()


def test_separate_rejects_silent_question_truncation(tokenizer):
    rows = [ContextRecord("evidence", (QARecord("question question", "answer"),), "squad")]
    with pytest.raises(ValueError, match="Question exceeds"):
        collate_context_records(rows, tokenizer, max_question_tokens=1, system_prompt_mode="separate")
