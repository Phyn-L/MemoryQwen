"""Prompt contents, configuration routing, and evaluation protocol isolation."""
import json
from types import SimpleNamespace

import pytest
import torch
import yaml
from transformers import BatchEncoding

from scripts.evaluation import test_icl_suite as suite
from src.icl_baseline import ICLExample, format_prompt, render_prompt, system_prompt_for
from src.qa_prompt import format_question
from utils.icl_config import load_icl_defaults


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return json.dumps(messages)

    def __call__(self, texts, **kwargs):
        return BatchEncoding({"input_ids": torch.tensor([[2, 3] for _ in texts])})

    def batch_decode(self, ids, **kwargs):
        return ["correct option"] * len(ids)


def _example(dataset="squad"):
    question = format_question(
        "Which option?", dataset,
        {"options": ["first option", "correct option", "third option", "fourth option"]},
    )
    return ICLExample("sample", dataset, "the context", question, ("correct option",))


@pytest.mark.parametrize("dataset", suite.DEFAULT_DATASETS)
def test_raw_contains_only_context_and_question_in_native_chat(dataset):
    tokenizer = _Tokenizer()
    example = _example(dataset)
    render_prompt(tokenizer, example, [], prompt_mode="raw")
    messages, options = tokenizer.calls[0]
    assert messages == [{"role": "user", "content": f"{example.context}\n\n{example.question}"}]
    assert options == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}


def test_instruct_is_default_and_keeps_system_and_demonstrations():
    tokenizer = _Tokenizer()
    example = _example()
    render_prompt(tokenizer, example, [example])
    messages, _ = tokenizer.calls[0]
    assert messages == [
        {"role": "system", "content": system_prompt_for(example.dataset)},
        {"role": "user", "content": format_prompt(example, [example])},
    ]
    assert "Example 1:" in messages[1]["content"]
    assert "Answer: correct option" in messages[1]["content"]


def test_raw_rejects_demonstrations_instead_of_silently_dropping_them():
    example = _example()
    with pytest.raises(ValueError, match="zero demonstrations"):
        render_prompt(_Tokenizer(), example, [example], prompt_mode="raw")


@pytest.mark.parametrize("mode", ["raw", "instruct"])
def test_chat_template_toggle_is_independent_of_prompt_mode(mode):
    tokenizer = _Tokenizer()
    example = _example()
    actual = render_prompt(tokenizer, example, [], use_chat_template=False, prompt_mode=mode)
    assert not tokenizer.calls
    expected = f"{example.context}\n\n{example.question}" if mode == "raw" else format_prompt(example, [])
    assert actual == expected


def test_unknown_prompt_mode_is_rejected():
    with pytest.raises(ValueError, match="raw or instruct"):
        render_prompt(_Tokenizer(), _example(), [], prompt_mode="typo")


@pytest.fixture
def model_path(tmp_path):
    path = tmp_path / "model"
    path.mkdir()
    (path / "config.json").write_text("{}")
    return str(path)


@pytest.mark.parametrize("mode,shots", [("raw", 0), ("instruct", 4)])
def test_worker_cli_selects_mode_and_defaults(model_path, mode, shots):
    args = suite.parse_args(["--model", model_path, "--prompt-mode", mode])
    assert args.prompt_mode == mode
    assert args.num_shots == shots
    assert not args.no_chat_template


def test_yaml_prompt_mode_and_cli_override(tmp_path, model_path):
    path = tmp_path / "icl.yaml"
    path.write_text(yaml.safe_dump({
        "model": {"name": model_path},
        "inference": {"prompt_mode": "raw", "num_shots": 0},
    }))
    defaults = load_icl_defaults(path)
    assert defaults["prompt_mode"] == "raw"
    assert not defaults["no_chat_template"]
    raw = suite.parse_args(["--config", str(path)])
    assert raw.prompt_mode == "raw" and raw.num_shots == 0
    instruct = suite.parse_args([
        "--config", str(path), "--prompt-mode", "instruct", "--num-shots", "2",
    ])
    assert instruct.prompt_mode == "instruct" and instruct.num_shots == 2


@pytest.mark.parametrize("inference", [
    {"prompt_mode": "invalid"},
    {"prompt_mode": "raw", "num_shots": 2},
])
def test_invalid_yaml_prompt_settings_fail(tmp_path, inference):
    path = tmp_path / "icl.yaml"
    path.write_text(yaml.safe_dump({"inference": inference}))
    with pytest.raises(ValueError, match="prompt"):
        load_icl_defaults(path)


def test_raw_cli_rejects_few_shot(model_path):
    with pytest.raises(SystemExit):
        suite.parse_args(["--model", model_path, "--prompt-mode", "raw", "--num-shots", "1"])


@pytest.mark.parametrize("mode", ["raw", "instruct"])
def test_suite_forwards_mode_to_worker_and_summary(tmp_path, model_path, mode):
    dataset_dir = tmp_path / "data" / "1qa" / "meta" / "test"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "squad.json").write_text("{}\n")
    output = tmp_path / "output"
    suite.suite_main([
        "--model", model_path, "--datasets", "squad", "--data-root", str(tmp_path / "data"),
        "--output-dir", str(output), "--prompt-mode", mode, "--dry-run",
    ])
    row, = json.loads((output / "summary.json").read_text())
    command = row["command"]
    assert row["prompt_mode"] == mode
    assert command[command.index("--prompt-mode") + 1] == mode


@pytest.mark.parametrize("mode", ["raw", "instruct"])
def test_metadata_records_the_actual_prompt_protocol(monkeypatch, model_path, mode):
    args = suite.parse_args(["--model", model_path, "--prompt-mode", mode, "--num-shots", "0"])
    monkeypatch.setattr(suite, "dataset_paths", lambda args: {"squad": "data.json"})
    protocol = suite.experiment_metadata(args, 1, {"squad": []})["protocol"]
    assert protocol["prompt_mode"] == mode
    assert protocol["chat_template"] is True
    if mode == "raw":
        assert protocol["system_prompts"] == {}
    else:
        assert protocol["system_prompts"]["squad"] == system_prompt_for("squad")


@pytest.mark.parametrize("mode", ["raw", "instruct"])
def test_evaluation_uses_selected_mode_and_writes_scores(tmp_path, mode):
    tokenizer = _Tokenizer()
    example = _example()
    args = SimpleNamespace(
        resume=False, batch_size=1, num_workers=0, max_new_tokens=8,
        no_chat_template=False, prompt_mode=mode, max_input_tokens=128,
    )

    class Model:
        def generate(self, input_ids, **kwargs):
            return torch.cat([input_ids, torch.tensor([[4]])], dim=1)

    suite.evaluate_dataset(
        args, "squad", [example], [], tokenizer, Model(), 0, 1, torch.device("cpu"), tmp_path,
    )
    messages, _ = tokenizer.calls[0]
    assert [message["role"] for message in messages] == (
        ["user"] if mode == "raw" else ["system", "user"]
    )
    metrics = json.loads((tmp_path / "squad.metrics.json").read_text())
    assert metrics["em"] == 1.0


def test_resume_cannot_mix_prompt_modes(tmp_path):
    path = tmp_path / "run_config.json"
    path.write_text(json.dumps({"protocol": {"prompt_mode": "instruct", "chat_template": True}}))
    args = SimpleNamespace(resume=True, output_dir=str(tmp_path), prompt_mode="raw", no_chat_template=False)
    with pytest.raises(ValueError, match="Cannot resume"):
        suite.validate_resume_prompt(args)
    args.prompt_mode = "instruct"
    suite.validate_resume_prompt(args)
    args.no_chat_template = True
    with pytest.raises(ValueError, match="Cannot resume"):
        suite.validate_resume_prompt(args)
