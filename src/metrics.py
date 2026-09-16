"""Answer-level QA metrics: one implementation, two normalizations.

The training-time evaluator and the few-shot ICL baseline must be scored the same
way for their numbers to be comparable, but they previously carried two
independent copies of ``exact_match`` / ``token_f1`` / ``rouge_l`` that differed
in string normalization -- and two different quantities were both called ``bleu``.

* :func:`normalize_text` keeps articles. This is the evaluator's original
  normalizer, so existing run metrics keep their meaning.
* :func:`normalize_official` is the official SQuAD normalizer (lowercase, strip
  punctuation, drop articles). The ICL baseline reports official numbers, so it
  passes this one.

Every metric takes a ``normalize`` callable, so both stacks share the arithmetic
and differ only in the normalizer they choose.
"""

from collections import Counter
import re
import string


# The four answer-level metrics, in the order every consumer must use them. The
# distributed reduction flattens them into a list, so a key appearing in a different
# order on different ranks would silently mix values up; keeping the canonical order here
# (and asserting it in tests/test_metrics.py) is what makes that impossible.
LEGACY_METRIC_KEYS: tuple[str, ...] = ("em", "f1", "rouge_l", "precision")
OFFICIAL_SUFFIX = "_official"
METRIC_KEYS: tuple[str, ...] = LEGACY_METRIC_KEYS + tuple(
    f"{key}{OFFICIAL_SUFFIX}" for key in LEGACY_METRIC_KEYS
)


def normalize_text(value) -> str:
    """Lowercase, replace non-alphanumerics with spaces, collapse whitespace."""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(value).lower()).split())


def normalize_official(value) -> str:
    """Official SQuAD-style lowercase/punctuation/article normalization."""
    text = str(value).lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, reference, normalize=normalize_text) -> float:
    return float(normalize(prediction) == normalize(reference))


def token_f1(prediction, reference, normalize=normalize_text) -> float:
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


def rouge_l(prediction, reference, normalize=normalize_text) -> float:
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


def unigram_precision(prediction, reference, normalize=normalize_text) -> float:
    """Fraction of predicted tokens that also occur in the reference.

    Formerly named ``bleu``, which it is not: BLEU needs n-gram precisions and a
    brevity penalty (the real thing is ``icl_baseline.corpus_bleu``).
    """
    prediction_tokens = normalize(prediction).split()
    reference_tokens = normalize(reference).split()
    if not prediction_tokens or not reference_tokens:
        return 0.0
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    return overlap / len(prediction_tokens)


def qa_metrics(prediction, reference, normalize=normalize_text) -> dict[str, float]:
    return {
        "em": exact_match(prediction, reference, normalize),
        "f1": token_f1(prediction, reference, normalize),
        "rouge_l": rouge_l(prediction, reference, normalize),
        "precision": unigram_precision(prediction, reference, normalize),
    }


def qa_metrics_official(prediction, reference) -> dict[str, float]:
    """The same metrics under the official SQuAD normalizer, suffixed ``_official``."""
    return {
        f"{key}{OFFICIAL_SUFFIX}": value
        for key, value in qa_metrics(prediction, reference, normalize_official).items()
    }


def qa_metrics_all(prediction, reference) -> dict[str, float]:
    """Both normalizations at once, in :data:`METRIC_KEYS` order.

    The training-time evaluator has always scored with :func:`normalize_text` (which keeps
    articles), while the published ICL baseline scores with :func:`normalize_official`.
    Reporting both means existing run metrics keep their meaning *and* the baseline
    comparison is exact, instead of asking the reader to assume the two rulers agree --
    they do not: ``prediction="a cat", reference="cat"`` is ``em=0, f1=0.667`` under the
    legacy ruler and ``em=1, f1=1`` under the official one.
    """
    values = qa_metrics(prediction, reference)
    values.update(qa_metrics_official(prediction, reference))
    return {key: values[key] for key in METRIC_KEYS}
