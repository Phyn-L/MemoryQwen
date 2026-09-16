from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .metrics import METRIC_KEYS, best_reference_metrics, normalize_answer


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
    """Yield ``(flat_row, line_number, index_within_line)`` for either file schema.

    ``contexts/standardized/`` stores one question per line. ``contexts/aggregated/`` groups
    a context's questions under ``qa_pairs``, and the per-question fields are identical (also
    for RACE, whose ``metadata.options``/``answer_letter`` survive the aggregation). Exploding
    the nested schema here means every consumer reads exactly one layout, which is what lets
    the baseline and the training evaluator share a data tree instead of two files that merely
    happen to agree today.
    """
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            pairs = row.get("qa_pairs")
            if pairs is None:
                yield row, line_number, 0
                continue
            for index, pair in enumerate(pairs or ()):
                flat = dict(pair)
                flat["context"] = row.get("context", "")
                flat.setdefault("dataset", row.get("dataset", dataset))
                yield flat, line_number, index


def load_jsonl(path: str | Path, dataset: str) -> list[ICLExample]:
    """Load an evaluation file, skipping questions that have no gold answer.

    The aggregated files are SQuAD v2.0-shaped, i.e. they also carry the unanswerable
    questions; the training pipeline drops those (`filter_no_qa`), so scoring them here would
    compare the baseline against a different question set. Skipping them keeps the two
    harnesses on exactly the same questions.
    """
    records = []
    for row, line_number, index in iter_examples(path, dataset):
        example = _from_row(row, dataset, line_number, index)
        if not example.references:
            continue
        records.append(example)
    return records


def sample_jsonl(path: str | Path, dataset: str, count: int, seed: int) -> list[ICLExample]:
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
        raise ValueError(f"Requested {count} demonstrations from {path}, found {len(reservoir)}")
    return reservoir


def _from_row(row: dict, dataset: str, line_number: int, index: int = 0) -> ICLExample:
    references = row.get("answers", [])
    if isinstance(references, str):
        references = [references]
    references = tuple(
        str(value).strip() for value in references
        if value is not None and str(value).strip()
    )
    metadata = row.get("metadata") or {}
    options = tuple(str(value).strip() for value in metadata.get("options", []))
    letter = str(metadata.get("answer_letter", "")).strip().upper()
    # ``index`` disambiguates the several questions that share one aggregated context line.
    identifier = str(row.get("id") or f"{dataset}-{line_number}-{index}")
    return ICLExample(
        id=identifier, dataset=dataset, context=str(row["context"]).strip(),
        question=str(row["question"]).strip(), references=references,
        options=options, answer_letter=letter,
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
        blocks.append(f"Example {index}:\n{_question_block(demo)}\nAnswer: {_demo_answer(demo)}")
    blocks.append(f"Now answer this question:\n{_question_block(example)}\nAnswer:")
    return "\n\n".join(blocks)


def render_prompt(tokenizer, example: ICLExample, demonstrations: Sequence[ICLExample], use_chat_template: bool = True) -> str:
    prompt = format_prompt(example, demonstrations)
    if not use_chat_template or not getattr(tokenizer, "apply_chat_template", None):
        return prompt
    messages = [
        {"role": "system", "content": "You are a precise question-answering system."},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _question_block(example: ICLExample) -> str:
    block = f"Passage:\n{example.context}\n\nQuestion: {example.question}"
    if example.dataset == "race":
        if len(example.options) != 4:
            raise ValueError(f"RACE example {example.id} has {len(example.options)} options, expected 4")
        choices = "\n".join(f"{letter}. {option}" for letter, option in zip("ABCD", example.options))
        block += f"\nOptions:\n{choices}"
    return block


def _demo_answer(example: ICLExample) -> str:
    if example.dataset == "race":
        return example.answer_letter
    return example.references[0] if example.references else ""


def parse_prediction(example: ICLExample, raw_prediction: str) -> tuple[str, str]:
    raw = raw_prediction.strip()
    if example.dataset != "race":
        first_line = raw.splitlines()[0].strip() if raw else ""
        return first_line, ""
    match = re.search(r"(?:^|[\s(\[])s*([A-D])(?:[\s).:\]]|$)", raw.upper())
    if not match:
        normalized = normalize_answer(raw)
        for index, option in enumerate(example.options):
            if normalized == normalize_answer(option):
                letter = "ABCD"[index]
                return option, letter
        return raw, ""
    letter = match.group(1)
    return example.options["ABCD".index(letter)], letter


def example_metrics(prediction: str, references: Sequence[str]) -> dict[str, float]:
    """Official normalization and best-reference reduction, shared with the evaluator.

    This is a thin alias of :func:`src.metrics.best_reference_metrics` so the baseline and
    the memory model cannot drift apart in either the normalizer or the reduction.
    """
    return best_reference_metrics(prediction, references)


def corpus_bleu(predictions: Sequence[str], references: Sequence[Sequence[str]], max_order: int = 4) -> float:
    """Corpus BLEU with closest-reference length and add-one smoothing."""
    clipped = [0] * max_order
    totals = [0] * max_order
    predicted_length = reference_length = 0
    for prediction, sample_references in zip(predictions, references):
        prediction_tokens = normalize_answer(prediction).split()
        reference_tokens = [normalize_answer(value).split() for value in (sample_references or [""])]
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
                    maximum_reference_counts[ngram] = max(maximum_reference_counts[ngram], count)
            clipped[order - 1] += sum(
                min(count, maximum_reference_counts[ngram]) for ngram, count in prediction_counts.items()
            )
            totals[order - 1] += sum(prediction_counts.values())
    if predicted_length == 0:
        return 0.0
    precisions = [(match + 1) / (total + 1) for match, total in zip(clipped, totals)]
    brevity_penalty = 1.0 if predicted_length > reference_length else math.exp(1 - reference_length / predicted_length)
    return brevity_penalty * math.exp(sum(math.log(value) for value in precisions) / max_order)


def _ngrams(tokens: Sequence[str], order: int) -> Counter:
    return Counter(tuple(tokens[index:index + order]) for index in range(len(tokens) - order + 1))


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
        result["accuracy"] = sum(row.get("predicted_letter") == row.get("answer_letter") for row in rows) / len(rows)
    return result
