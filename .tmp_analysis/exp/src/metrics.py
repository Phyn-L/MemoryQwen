from collections import Counter
import re


def normalize_text(value):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(value).lower()).split())


def exact_match(prediction, reference):
    return float(normalize_text(prediction) == normalize_text(reference))


def token_f1(prediction, reference):
    prediction_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if not prediction_tokens or not reference_tokens or not overlap:
        return float(not prediction_tokens and not reference_tokens)
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs(first, second):
    row = [0] * (len(second) + 1)
    for token in first:
        old = 0
        for index, other in enumerate(second, 1):
            previous = row[index]
            row[index] = old + 1 if token == other else max(row[index], row[index - 1])
            old = previous
    return row[-1]


def rouge_l(prediction, reference):
    prediction_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()
    length = _lcs(prediction_tokens, reference_tokens)
    if not length:
        return 0.0
    precision = length / len(prediction_tokens)
    recall = length / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu(prediction, reference):
    prediction_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()
    if not prediction_tokens or not reference_tokens:
        return 0.0
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    return overlap / len(prediction_tokens)


def qa_metrics(prediction, reference):
    rouge = rouge_l(prediction, reference)
    return {
        "em": exact_match(prediction, reference),
        "f1": token_f1(prediction, reference),
        "rouge_l": rouge,
        "bleu": bleu(prediction, reference),
    }
