from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .metrics import METRIC_KEYS, answer_line, best_reference_metrics, normalize_answer

DEFAULT_SYSTEM_PROMPT = "You are a precise question-answering system. Use only the provided passage and follow the requested output format exactly."

SYSTEM_PROMPTS = {
    "squad": (
        "Use only the provided passage to answer the question. Return the shortest supported "
        "answer span, with no explanation or extra formatting."
    ),
    "ms_marco_v1": (
        "Use only the provided passage to answer the question. Give a concise, direct answer "
        "that is supported by the passage, with no explanation or extra formatting."
    ),
    "ms_marco_v2": (
        "Use only the provided passage to answer the question. Synthesize the most direct "
        "answer supported by the passage, with no explanation or extra formatting."
    ),
    "hotpotqa": (
        "Use all relevant parts of the provided passage to reason about the question. "
        "Combine evidence when needed and return only a concise answer, with no explanation "
        "or extra formatting."
    ),
    "race": (
        "Use only the provided passage to solve the multiple-choice question. Choose the "
        "single best option and return only its uppercase letter: A, B, C, or D."
    ),
}

_MS_MARCO_PROMPT_KEYS = {
    "ms_marco_v1_1": "ms_marco_v1",
    "ms_marco_v2_1": "ms_marco_v2",
}


@dataclass(frozen=True)
class ICLExample:
    id: str
    dataset: str
    context: str
    question: str
    references: tuple[str, ...]
    options: tuple[str, ...] = ()
    answer_letter: str = ""


def iter_examples(path: str | Path, dataset: str):
    """Yield ``(flat_row, line_number, index_within_line)`` from an aggregated file.

    Every file uses one context and a ``conversations`` list. Each user message is paired
    with the following assistant message and yielded as one evaluation example.
    """
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            conversations = row.get("conversations")
            if not isinstance(conversations, list):
                raise ValueError(
                    f"{path}:{line_number} has no 'conversations'; expected the conversation schema"
                )
            pending = None
            index = 0
            for message in conversations:
                role = str(message.get("role", "")).strip().lower()
                if role == "user":
                    pending = message
                elif role == "assistant" and pending is not None:
                    references = message.get("answers", [message.get("content", "")])
                    if isinstance(references, str):
                        references = [references]
                    flat = {
                        "question": pending.get("content", ""),
                        "answers": references,
                        "context": row.get("context", ""),
                        "dataset": row.get("dataset", dataset),
                        "context_id": row.get("context_id", row.get("id", "")),
                    }
                    metadata = dict(message.get("metadata") or {})
                    if message.get("answer_letter"):
                        metadata["answer_letter"] = message["answer_letter"]
                    flat["metadata"] = metadata
                    yield flat, line_number, index
                    index += 1
                    pending = None


def load_jsonl(
    path: str | Path, dataset: str, source_version: str | None = None
) -> list[ICLExample]:
    """Load an evaluation file, skipping questions that have no gold answer.

    The aggregated files are SQuAD v2.0-shaped, i.e. they also carry the unanswerable
    questions; the training pipeline drops those (`filter_no_qa`), so scoring them here would
    compare the baseline against a different question set. Skipping them keeps the two
    harnesses on exactly the same questions.
    """
    records = []
    for row, line_number, index in iter_examples(path, dataset):
        if source_version and str(row.get("source_dataset", "")) != source_version:
            continue
        example = _from_row(row, dataset, line_number, index)
        if not example.references:
            continue
        records.append(example)
    return records


def limit_examples(records, max_contexts=None, max_context_tokens=None, tokenizer=None):
    """Trim a split the way the memory evaluator's validation set is trimmed.

    The two harnesses must score the same *items*, and by default they did not: the memory
    side reads the cached validation split, which drops contexts longer than
    ``data.max_context_tokens`` and then keeps at most ``data.validation_max_samples``
    **contexts**, while this script read the raw file and capped **examples** (QA rows) with
    ``--max-samples``. Same normalizer, same reduction, different question set.

    ``max_contexts`` keeps the examples of the first N distinct contexts in file order and
    ``max_context_tokens`` drops the examples whose context does not fit, so
    ``--max-contexts`` / ``--max-context-tokens`` can reproduce the memory side's subset.
    """
    limited = list(records)
    if max_context_tokens is not None:
        if tokenizer is None:
            raise ValueError("max_context_tokens needs a tokenizer")
        limited = [
            example
            for example in limited
            if len(tokenizer(example.context, add_special_tokens=False).input_ids)
            <= int(max_context_tokens)
        ]
    if max_contexts is not None:
        seen: list[str] = []
        kept = []
        for example in limited:
            key = example.context
            if key not in seen:
                if len(seen) >= int(max_contexts):
                    continue
                seen.append(key)
            kept.append(example)
        limited = kept
    return limited


def sample_context_rows(records, max_contexts=None, qa_per_context=None, seed=42):
    """Shuffle context groups, keep N groups, and keep up to K QA rows per group."""
    if max_contexts is None and qa_per_context is None:
        return records
    groups = {}
    order = []
    for record in records:
        if record.context not in groups:
            groups[record.context] = []
            order.append(record.context)
        groups[record.context].append(record)
    rng = random.Random(seed)
    rng.shuffle(order)
    if max_contexts is not None:
        order = order[:max_contexts]
    sampled = []
    for context in order:
        rows = groups[context]
        sampled.extend(rows if qa_per_context is None else rows[:qa_per_context])
    return sampled


def sample_jsonl(
    path: str | Path, dataset: str, count: int, seed: int
) -> list[ICLExample]:
    """Reservoir-sample demonstrations without materializing the train split."""
    if count <= 0:
        return []
    rng = random.Random(seed)
    reservoir: list[ICLExample] = []
    seen = 0
    for row, line_number, index in iter_examples(path, dataset):
        example = _from_row(row, dataset, line_number, index)
        if not example.references or (dataset == "race" and not example.answer_letter):
            continue
        seen += 1
        if len(reservoir) < count:
            reservoir.append(example)
        else:
            position = rng.randrange(seen)
            if position < count:
                reservoir[position] = example
    if len(reservoir) < count:
        raise ValueError(
            f"Requested {count} demonstrations from {path}, found {len(reservoir)}"
        )
    return reservoir


def _from_row(row: dict, dataset: str, line_number: int, index: int = 0) -> ICLExample:
    references = row.get("answers", [])
    if isinstance(references, str):
        references = [references]
    references = tuple(
        str(value).strip()
        for value in references
        if value is not None and str(value).strip()
    )
    metadata = row.get("metadata") or {}
    options = tuple(str(value).strip() for value in metadata.get("options", []))
    letter = str(metadata.get("answer_letter", "")).strip().upper()
    # ``index`` disambiguates the several questions that share one aggregated context line.
    identifier = str(row.get("id") or f"{dataset}-{line_number}-{index}")
    # The aggregated MS MARCO context is constructed upstream from passages with
    # is_selected=1. Keep the ICL path on that canonical context so it matches the
    # MemoryQwen evaluator and the selected-passage protocol used by this project.
    context = str(row["context"]).strip()
    return ICLExample(
        id=identifier,
        dataset=dataset,
        context=context,
        question=str(row["question"]).strip(),
        references=references,
        options=options,
        answer_letter=letter,
    )


def format_prompt(example: ICLExample, demonstrations: Sequence[ICLExample]) -> str:
    if example.dataset == "race":
        instruction = (
            "Read each passage and answer its multiple-choice question. "
            "Return only the option letter (A, B, C, or D)."
        )
    else:
        instruction = (
            "Answer each question using only the given passage. Return only a short answer, "
            "without explanation."
        )
    blocks = [instruction]
    for index, demo in enumerate(demonstrations, 1):
        blocks.append(
            f"Example {index}:\n{_question_block(demo)}\nAnswer: {_demo_answer(demo)}"
        )
    blocks.append(f"Now answer this question:\n{_question_block(example)}\nAnswer:")
    return "\n\n".join(blocks)


def system_prompt_for(dataset: str, source_version: str | None = None) -> str:
    """Resolve the dataset-specific chat system prompt.

    MS MARCO v1.1 and v2.1 share the on-disk ``ms_marco`` directory, so their source
    version is part of prompt selection. Other datasets use their directory name directly.
    Unknown datasets retain a strict generic QA prompt instead of failing evaluation.
    """
    prompt_key = (
        _MS_MARCO_PROMPT_KEYS.get(str(source_version), dataset)
        if dataset == "ms_marco"
        else dataset
    )
    return SYSTEM_PROMPTS.get(prompt_key, DEFAULT_SYSTEM_PROMPT)


def render_prompt(
    tokenizer,
    example: ICLExample,
    demonstrations: Sequence[ICLExample],
    use_chat_template: bool = True,
    source_version: str | None = None,
) -> str:
    prompt = format_prompt(example, demonstrations)
    if not use_chat_template or not getattr(tokenizer, "apply_chat_template", None):
        return prompt
    messages = [
        {
            "role": "system",
            "content": system_prompt_for(example.dataset, source_version),
        },
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def _question_block(example: ICLExample) -> str:
    block = f"Passage:\n{example.context}\n\nQuestion: {example.question}"
    if example.dataset == "race":
        if len(example.options) != 4:
            raise ValueError(
                f"RACE example {example.id} has {len(example.options)} options, expected 4"
            )
        choices = "\n".join(
            f"{letter}. {option}" for letter, option in zip("ABCD", example.options)
        )
        block += f"\nOptions:\n{choices}"
    return block


def _demo_answer(example: ICLExample) -> str:
    if example.dataset == "race":
        return example.answer_letter
    return example.references[0] if example.references else ""


def parse_prediction(example: ICLExample, raw_prediction: str) -> tuple[str, str]:
    raw = raw_prediction.strip()
    if example.dataset != "race":
        # The very same helper the memory evaluator uses, so the two harnesses cannot drift
        # apart on what "the prediction" is.
        return answer_line(raw), ""
    return parse_race_prediction(raw, example.options)


def parse_race_prediction(raw_prediction: str, options: Sequence[str] = ()) -> tuple[str, str]:
    """Return the answer text and option letter from a RACE generation."""
    raw = raw_prediction.strip()
    match = re.search(r"(?:^|[\s(\[])\s*([A-D])(?:[\s).:\]]|$)", raw.upper())
    if not match:
        normalized = normalize_answer(raw)
        for index, option in enumerate(options):
            if normalized == normalize_answer(option):
                return option, "ABCD"[index]
        return raw, ""
    letter = match.group(1)
    option = options["ABCD".index(letter)] if len(options) == 4 else ""
    return option or letter, letter


def example_metrics(prediction: str, references: Sequence[str]) -> dict[str, float]:
    """Official normalization and best-reference reduction, shared with the evaluator.

    This is a thin alias of :func:`src.metrics.best_reference_metrics` so the baseline and
    the memory model cannot drift apart in either the normalizer or the reduction.
    """
    return best_reference_metrics(prediction, references)


def corpus_bleu(
    predictions: Sequence[str], references: Sequence[Sequence[str]], max_order: int = 4
) -> float:
    """Corpus BLEU with closest-reference length and add-one smoothing."""
    clipped = [0] * max_order
    totals = [0] * max_order
    predicted_length = reference_length = 0
    for prediction, sample_references in zip(predictions, references):
        prediction_tokens = normalize_answer(prediction).split()
        reference_tokens = [
            normalize_answer(value).split() for value in (sample_references or [""])
        ]
        predicted_length += len(prediction_tokens)
        reference_length += min(
            (len(value) for value in reference_tokens),
            key=lambda length: (abs(length - len(prediction_tokens)), length),
        )
        for order in range(1, max_order + 1):
            prediction_counts = _ngrams(prediction_tokens, order)
            maximum_reference_counts: Counter = Counter()
            for value in reference_tokens:
                counts = _ngrams(value, order)
                for ngram, count in counts.items():
                    maximum_reference_counts[ngram] = max(
                        maximum_reference_counts[ngram], count
                    )
            clipped[order - 1] += sum(
                min(count, maximum_reference_counts[ngram])
                for ngram, count in prediction_counts.items()
            )
            totals[order - 1] += sum(prediction_counts.values())
    if predicted_length == 0:
        return 0.0
    precisions = [(match + 1) / (total + 1) for match, total in zip(clipped, totals)]
    brevity_penalty = (
        1.0
        if predicted_length > reference_length
        else math.exp(1 - reference_length / predicted_length)
    )
    return brevity_penalty * math.exp(
        sum(math.log(value) for value in precisions) / max_order
    )


def _ngrams(tokens: Sequence[str], order: int) -> Counter:
    return Counter(
        tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1)
    )


def aggregate_metrics(rows: Iterable[dict]) -> dict[str, float | int]:
    rows = list(rows)
    if not rows:
        return {"count": 0, "bleu_4": 0.0, **{key: 0.0 for key in METRIC_KEYS}}
    result: dict[str, float | int] = {"count": len(rows)}
    for metric in METRIC_KEYS:
        result[metric] = sum(float(row[metric]) for row in rows) / len(rows)
    result["bleu_4"] = corpus_bleu(
        [str(row["prediction"]) for row in rows],
        [list(row["references"]) for row in rows],
    )
    if rows[0].get("dataset") == "race":
        result["accuracy"] = sum(
            row.get("predicted_letter") == row.get("answer_letter") for row in rows
        ) / len(rows)
    return result
