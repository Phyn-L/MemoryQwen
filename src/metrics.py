"""Answer-level QA metrics: one implementation, one normalizer, one reduction.

The training-time evaluator and the few-shot ICL baseline must produce the *same*
number for the same prediction, otherwise comparing them is meaningless. They used to
carry independent copies that differed in two ways -- the string normalizer (articles
kept vs the official SQuAD normalizer) and the reduction over multiple gold answers
(first reference vs max over all references). Both differences are now gone: the
arithmetic, the normalizer and the multi-reference reduction all live here.

The normalizer is the official SQuAD one (lowercase, punctuation stripped, articles
dropped). It is exposed as the ``normalize`` argument rather than hard-coded, so a future
variant can be introduced explicitly instead of by editing the metric bodies.

The scored set is ``em``, ``f1`` and ``rouge_l`` (see ``METRIC_KEYS``); ``f1`` is the
official SQuAD token-level F1, so a number from here is directly comparable with published
SQuAD results and with ``icl_baseline``.
"""

from collections import Counter
from typing import Sequence
import re
import string


# The answer-level metrics, in the order every consumer must use them. The distributed
# reduction flattens them into a list, so a key appearing in a different order on
# different ranks would silently mix values up; keeping the canonical order here (and
# asserting it in tests/test_metrics.py) is what makes that impossible.
#
# ``precision`` deliberately is not here. It was unigram precision over the prediction
# (``unigram_precision``), which rewards short answers and has no recall or brevity
# penalty: a one-token prediction that happens to occur in the reference scores 1.0,
# and a fully correct but longer answer scores below it. It also moved almost line for
# line with ``f1`` on this task while being the only metric that a truncated answer can
# game, so it added a number without adding a signal. ``f1`` (the harmonic mean of the
# same overlap) already covers that overlap and stays comparable to the SQuAD
# literature. The real BLEU lives in ``icl_baseline.corpus_bleu``.
METRIC_KEYS: tuple[str, ...] = ("em", "f1", "rouge_l")


def normalize_answer(value) -> str:
    """Official SQuAD-style lowercase/punctuation/article normalization."""
    text = str(value).lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, reference, normalize=normalize_answer) -> float:
    return float(normalize(prediction) == normalize(reference))


def token_f1(prediction, reference, normalize=normalize_answer) -> float:
    prediction_tokens = normalize(prediction).split()
    reference_tokens = normalize(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs(first, second) -> int:
    row = [0] * (len(second) + 1)
    for token in first:
        old = 0
        for index, other in enumerate(second, 1):
            previous = row[index]
            row[index] = old + 1 if token == other else max(row[index], row[index - 1])
            old = previous
    return row[-1]


def rouge_l(prediction, reference, normalize=normalize_answer) -> float:
    prediction_tokens = normalize(prediction).split()
    reference_tokens = normalize(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    length = _lcs(prediction_tokens, reference_tokens)
    if not length:
        return 0.0
    precision = length / len(prediction_tokens)
    recall = length / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_metrics(prediction, reference, normalize=normalize_answer) -> dict[str, float]:
    """The scored metrics against a single gold answer."""
    return {
        "em": exact_match(prediction, reference, normalize),
        "f1": token_f1(prediction, reference, normalize),
        "rouge_l": rouge_l(prediction, reference, normalize),
    }


def best_reference_metrics(prediction: str, references: Sequence[str]) -> dict[str, float]:
    """Metric-wise max over the gold answers -- the official SQuAD reduction.

    SQuAD ships several annotator answers per question and a prediction counts as correct
    if it matches *any* of them; taking the max per metric (rather than one reference's
    score) is what the published numbers do. Both the training evaluator and the ICL
    baseline call this, so neither the normalizer nor the reduction can drift apart again.
    """
    references = tuple(references) or ("",)
    scored = [qa_metrics(prediction, reference) for reference in references]
    return {key: max(row[key] for row in scored) for key in METRIC_KEYS}
