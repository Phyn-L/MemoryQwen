from __future__ import annotations

import json
import math
import random
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .metrics import METRIC_KEYS, answer_line, best_reference_metrics, normalize_answer
from .qa_prompt import format_question

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise question-answering system. Use only the provided passage and follow "
    "the requested output format exactly."
)

# Keyed by the dataset name used everywhere else now (src/eval_data.py). The old aggregated
# corpus shared one ``ms_marco`` directory between v1.1 and v2.1 and needed a source-version
# argument to pick the prompt; the 1QA build writes one file per dataset, so the name decides.
SYSTEM_PROMPTS = {
    "squad": (
        "Use only the provided passage to answer the question. Return the shortest supported "
        "answer span, with no explanation or extra formatting."
    ),
    "msmarcov1.1": (
        "Use only the provided passage to answer the question. Give a concise, direct answer "
        "that is supported by the passage, with no explanation or extra formatting."
    ),
    "msmarcov2.1": (
        "Use only the provided passage to answer the question. Synthesize the most direct "
        "answer supported by the passage, with no explanation or extra formatting."
    ),
    "hotpotqa(distractor)": (
        "Use all relevant parts of the provided passage to reason about the question. "
        "Combine evidence when needed and return only a concise answer, with no explanation "
        "or extra formatting."
    ),
}


@dataclass(frozen=True)
class ICLExample:
    id: str
    dataset: str
    context: str
    question: str
    references: tuple[str, ...]


def iter_examples(path: str | Path, dataset: str | None = None):
    """Yield ``(flat_row, line_number, index_within_line)`` from a JSONL file.

    Two schemas come out of ``scripts/data/`` and both are accepted here:

    * the per-dataset evaluation files hold one context per line with a ``conversations``
      list (each user message paired with the assistant message that follows it);
    * the shared training file holds one context per line with a ``qa_pairs`` list, and all
      datasets in the same file -- ``dataset`` then filters the rows, which is how the ICL
      baseline samples its demonstrations without materializing a 1.15M-row split per dataset.
    """
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_dataset = str(row.get("dataset", ""))
            if dataset and row_dataset and row_dataset != dataset:
                continue
            pairs = row.get("qa_pairs")
            if isinstance(pairs, list):
                for index, pair in enumerate(pairs):
                    if not isinstance(pair, dict):
                        continue
                    flat = _flat_row_from_pair(row, pair, dataset)
                    if flat is not None:
                        yield flat, line_number, index
                continue
            conversations = row.get("conversations")
            if not isinstance(conversations, list):
                raise ValueError(
                    f"{path}:{line_number} has neither 'conversations' nor 'qa_pairs'; "
                    "expected a scripts/data JSONL row"
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
                        "dataset": row_dataset or dataset or "",
                        "context_id": row.get("context_id", row.get("id", "")),
                    }
                    flat["metadata"] = dict(message.get("metadata") or {})
                    yield flat, line_number, index
                    index += 1
                    pending = None


def _flat_row_from_pair(row: dict, pair: dict, dataset: str | None):
    """One ``qa_pairs`` entry in the flat shape ``_from_row``/``sample_jsonl`` expect."""
    if not isinstance(pair.get("metadata"), dict):
        pair = {**pair, "metadata": row.get("metadata") or {}}
    question = str(pair.get("question", "")).strip()
    if not question:
        return None
    question = format_question(
        question, str(row.get("dataset", "")) or (dataset or ""), pair["metadata"],
    )
    references = pair.get("answers") or ()
    if isinstance(references, str):
        references = (references,)
    references = [value for value in references if str(value).strip()]
    answer = str(pair.get("answer") or (references[0] if references else "")).strip()
    if not references and answer:
        references = [answer]
    if not references:
        return None
    return {
        "question": question,
        "answers": references,
        "context": row.get("context", ""),
        "dataset": str(row.get("dataset", "")) or (dataset or ""),
        "context_id": row.get("context_id", row.get("id", "")),
        "metadata": dict(pair.get("metadata") or {}),
        "id": str(pair.get("id") or row.get("id") or ""),
    }


def load_jsonl(path: str | Path, dataset: str) -> list[ICLExample]:
    """Load an evaluation file, skipping questions that have no gold answer.

    The source files are SQuAD v2.0-shaped, i.e. they also carry the unanswerable questions;
    the training pipeline drops those (`filter_no_qa`), so scoring them here would compare the
    baseline against a different question set. Skipping them keeps the two harnesses on
    exactly the same questions.
    """
    records = []
    for row, line_number, index in iter_examples(path, dataset):
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


def sample_context_rows(
    records, max_contexts=None, max_qa_sample_per_context=None, seed=42,
):
    """Shuffle context groups, keep N groups, and keep up to K QA rows per group."""
    if max_contexts is None and max_qa_sample_per_context is None:
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
        sampled.extend(
            rows if max_qa_sample_per_context is None
            else rows[:max_qa_sample_per_context]
        )
    return sampled


def sample_jsonl_many(
    path: str | Path, datasets: Sequence[str], count: int, seed: int = 42
) -> dict[str, list[ICLExample]]:
    """Reservoir-sample demonstrations for several datasets in **one** pass.

    The 1QA training split holds every dataset (1.15M contexts, several GB), so sampling it
    once per dataset would read it once per dataset. Each dataset keeps its own reservoir and
    its own RNG -- seeded from ``(seed, dataset)`` so a dataset's demonstrations do not depend
    on which other datasets were requested -- and ``count == 0`` returns before opening the
    file, which is the common zero-shot case.
    """
    datasets = list(datasets)
    if count <= 0 or not datasets:
        return {dataset: [] for dataset in datasets}
    reservoirs: dict[str, list[ICLExample]] = {dataset: [] for dataset in datasets}
    rngs = {dataset: random.Random(f"{seed}:{dataset}") for dataset in datasets}
    seen = {dataset: 0 for dataset in datasets}
    for row, line_number, index in iter_examples(path, None):
        dataset = str(row.get("dataset", ""))
        if dataset not in reservoirs:
            continue
        example = _from_row(row, dataset, line_number, index)
        if not example.references:
            continue
        seen[dataset] += 1
        reservoir = reservoirs[dataset]
        if len(reservoir) < count:
            reservoir.append(example)
        else:
            position = rngs[dataset].randrange(seen[dataset])
            if position < count:
                reservoir[position] = example
    short = sorted(dataset for dataset, rows in reservoirs.items() if len(rows) < count)
    if short:
        raise ValueError(
            f"Requested {count} demonstrations from {path}, found fewer for {short}"
        )
    return reservoirs


def _from_row(row: dict, dataset: str, line_number: int, index: int = 0) -> ICLExample:
    references = row.get("answers", [])
    if isinstance(references, str):
        references = [references]
    references = tuple(
        str(value).strip()
        for value in references
        if value is not None and str(value).strip()
    )
    # ``index`` disambiguates the several questions that share one context line.
    identifier = str(row.get("id") or f"{dataset}-{line_number}-{index}")
    # The MS MARCO context is constructed upstream from passages with is_selected=1. Keeping
    # the ICL path on that canonical context is what makes it match the MemoryQwen evaluator
    # and the selected-passage protocol used by this project.
    context = str(row["context"]).strip()
    return ICLExample(
        id=identifier,
        dataset=dataset,
        context=context,
        question=str(row["question"]).strip(),
        references=references,
    )


def format_prompt(example: ICLExample, demonstrations: Sequence[ICLExample]) -> str:
    instruction = (
        "Answer each question using only the given passage. Return only a short answer, "
        "without explanation."
    )
    blocks = [instruction]
    for index, demo in enumerate(demonstrations, 1):
        blocks.append(
            f"Example {index}:\n{_question_block(demo)}\nAnswer: {_demo_answer(demo)}"
        )
    suffix = "" if example.dataset == "race" else "\nAnswer:"
    blocks.append(f"Now answer this question:\n{_question_block(example)}{suffix}")
    return "\n\n".join(blocks)


def system_prompt_for(dataset: str) -> str:
    """Dataset-specific chat system prompt; unknown datasets keep the strict generic one."""
    return SYSTEM_PROMPTS.get(dataset, DEFAULT_SYSTEM_PROMPT)


def render_prompt(
    tokenizer,
    example: ICLExample,
    demonstrations: Sequence[ICLExample],
    use_chat_template: bool = True,
) -> str:
    prompt = format_prompt(example, demonstrations)
    if not use_chat_template or not getattr(tokenizer, "apply_chat_template", None):
        return prompt
    messages = [
        {
            "role": "system",
            "content": system_prompt_for(example.dataset),
        },
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _question_block(example: ICLExample) -> str:
    return f"Passage:\n{example.context}\n\nQuestion: {example.question}"


def _demo_answer(example: ICLExample) -> str:
    return example.references[0] if example.references else ""


def parse_prediction(example: ICLExample, raw_prediction: str) -> tuple[str, str]:
    return answer_line(raw_prediction.strip()), ""


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
    for prediction, sample_references in zip(predictions, references, strict=True):
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
    precisions = [
        (match + 1) / (total + 1) for match, total in zip(clipped, totals, strict=True)
    ]
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
    return result
