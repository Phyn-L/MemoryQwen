import json

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from src.data import (
    AggregatedContextDataset,
    ContextRecord,
    QARecord,
    TrainingTaskDataset,
    collate_context_records,
)


def _tokenizer():
    vocab = {"[PAD]": 0, "[UNK]": 1, "Q": 2, "A": 3, "R": 4, "foo": 5, "bar": 6}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]")


def test_input_context_limit_filters_before_training_collate(tmp_path):
    root = tmp_path / "squad"
    root.mkdir()
    rows = [
        {"context": "foo", "qa_pairs": [{"question": "Q", "answers": ["A"]}]},
        {"context": "foo bar foo", "qa_pairs": [{"question": "Q", "answers": ["A"]}]},
    ]
    (root / "train.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    dataset = AggregatedContextDataset(
        tmp_path, "squad", "train", _tokenizer(), max_context_tokens=8,
        max_input_context_length=2,
    )
    assert len(dataset) == 1


def test_mixed_task_rows_have_distinct_prompts_and_targets():
    tokenizer = _tokenizer()
    source = ContextRecord("foo bar", (QARecord("Q", "A"),))
    rows = [
        source.__class__(source.context, source.qa_pairs, task="qa"),
        source.__class__(source.context, source.qa_pairs, task="recon"),
    ]
    batch = collate_context_records(
        rows, tokenizer, max_context_tokens=8, max_question_tokens=8,
        max_answer_tokens=8, append_eos=False, training=True,
        qa_prompt="Q ", recon_prompt="R ",
    )
    assert batch["task_ids"].tolist() == [0, 1]
    assert batch["question_ids"].shape[1] > 1
    assert batch["answer_ids"].shape[1] == 2


def test_task_dataset_ratio_selects_qa_and_recon_candidates():
    source = [ContextRecord("foo", (QARecord("Q", "A"),)) for _ in range(10)]
    class Source:
        cache_metadata = {}
        def __len__(self): return len(source)
        def __getitem__(self, index): return source[index]
    dataset = TrainingTaskDataset(Source(), "mix", recon_ratio=0.25)
    indices = dataset.sample_indices(7)
    tasks = [dataset[index].task for index in indices]
    assert len(indices) == 10
    assert tasks.count("recon") == 2
    assert tasks.count("qa") == 8
