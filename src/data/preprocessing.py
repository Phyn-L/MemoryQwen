"""Filtering and token-length computation for context records."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor

from utils.ddp import is_main_process

from .records import ContextRecord, _messages_from_row


_FILTER_WORKER_TOKENIZER = None

_FILTER_WORKER_OPTIONS = None

_FILTER_WORKER_WITH_LENGTHS = False

_SORTISH_WORKER_TOKENIZER = None

def _init_filter_worker(tokenizer, options, with_lengths=False):
    global _FILTER_WORKER_TOKENIZER, _FILTER_WORKER_OPTIONS, _FILTER_WORKER_WITH_LENGTHS
    _FILTER_WORKER_TOKENIZER = tokenizer
    _FILTER_WORKER_OPTIONS = options
    _FILTER_WORKER_WITH_LENGTHS = with_lengths

def filter_rows_batch(rows, tokenizer, options, lengths_out: list[int] | None = None):
    """The one filter implementation; called in-process and inside the process pool.

    It is a module-level function of its arguments (rather than of worker globals) so the
    serial and the parallel driver run *the same* rules: the rules used to be written twice,
    once per driver, which is exactly how a filter silently drifts between them.
    """
    (
        max_context_tokens,
        filter_long_context,
        filter_no_conversation,
        max_question_tokens,
        max_answer_tokens,
        filter_qa_pair_answers,
    ) = options
    accepted = []
    for raw in rows:
        context = str(raw.get("context", "")).strip()
        if not context:
            continue
        messages = _messages_from_row(raw)
        if filter_no_conversation and not any(m.role == "assistant" for m in messages):
            continue
        turns = (
            ContextRecord("", messages, independent_qa=True).turns()
            if raw.get("qa_pairs")
            else ContextRecord("", messages).turns()
        )
        if not turns:
            continue
        if tokenizer is not None:
            length: int | None = None
            if filter_long_context:
                length = len(tokenizer(context, add_special_tokens=False,
                                       truncation=True, max_length=max_context_tokens + 1).input_ids)
                if not 0 < length <= max_context_tokens:
                    continue
            questions = [t.question for t in turns]
            qids = tokenizer(
                questions, add_special_tokens=False, truncation=True,
                max_length=max_question_tokens + 1,
            ).input_ids
            aids = tokenizer(
                [t.answer for t in turns], add_special_tokens=False, truncation=True,
                max_length=max_answer_tokens + 1,
            ).input_ids
            if any(
                not 0 < len(q) <= max_question_tokens
                or not 0 < len(a) <= max_answer_tokens
                for q, a in zip(qids, aids, strict=True)
            ) and (filter_qa_pair_answers or not raw.get("qa_pairs")):
                continue
        accepted.append(raw)
        if lengths_out is not None:
            # Every accepted row satisfies `length <= max_context_tokens`, so the truncated
            # count *is* the context's true token length (which is what the sortish sampler
            # buckets by). ``-1`` means "not measured" (no tokenizer / filter_long_context off).
            lengths_out.append(length if length is not None else -1)
    return accepted

def _filter_row_batch_in_worker(rows):
    """Process-pool entry point: read the tokenizer/options the initializer installed."""
    if not _FILTER_WORKER_WITH_LENGTHS:
        return filter_rows_batch(rows, _FILTER_WORKER_TOKENIZER, _FILTER_WORKER_OPTIONS)
    lengths: list[int] = []
    accepted = filter_rows_batch(
        rows, _FILTER_WORKER_TOKENIZER, _FILTER_WORKER_OPTIONS, lengths
    )
    return accepted, lengths

def filter_options(
    max_context_tokens, filter_long_context, filter_no_conversation,
    max_question_tokens, max_answer_tokens, filter_qa_pair_answers=False,
):
    """The rule set :func:`filter_rows_batch` expects, in one place.

    Kept as a tuple so it can also cross a process-pool boundary; ``scripts/data/build_cache.py``
    uses this to apply exactly the filter the loader applies.
    """
    return (
        max_context_tokens, filter_long_context, filter_no_conversation,
        max_question_tokens, max_answer_tokens,
        filter_qa_pair_answers,
    )

def _init_sortish_worker(tokenizer):
    global _SORTISH_WORKER_TOKENIZER
    _SORTISH_WORKER_TOKENIZER = tokenizer

def _sortish_length_batch(contexts):
    encoded = _SORTISH_WORKER_TOKENIZER(
        contexts,
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )
    input_ids = (
        encoded.input_ids
        if hasattr(encoded, "input_ids")
        else encoded["input_ids"]
    )
    if len(input_ids) != len(contexts):
        raise ValueError(
            "tokenizer returned a different number of rows for sortish lengths"
        )
    return [len(ids) for ids in input_ids]

def context_lengths(contexts, tokenizer, num_workers: int = 0, batch_size: int = 1024,
                    on_batch=None):
    """Token counts of ``contexts``, batched and in order.

    ``num_workers > 1`` sends the batches through the same pool the sortish sampler uses. Used
    by ``SortishSampler`` and available to ``scripts/data/build_cache.py`` (which gets its
    lengths straight out of the filter instead; see :func:`filter_rows_batch`).
    """
    lengths: list[int] = []
    pending: list[str] = []

    def flush():
        if not pending:
            return
        encoded = tokenizer(pending, add_special_tokens=False, truncation=False, padding=False)
        input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
        if len(input_ids) != len(pending):
            raise ValueError(
                "tokenizer returned a different number of rows for sortish lengths"
            )
        lengths.extend(len(ids) for ids in input_ids)
        if on_batch is not None:
            on_batch(len(pending))
        pending.clear()

    if num_workers <= 1:
        for context in contexts:
            pending.append(context)
            if len(pending) >= batch_size:
                flush()
        flush()
        return lengths

    def batches():
        batch = []
        for context in contexts:
            batch.append(context)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_sortish_worker,
        initargs=(tokenizer,),
    ) as executor:
        for batch_lengths in _bounded_process_map(
            executor, _sortish_length_batch, batches(), num_workers * 2
        ):
            lengths.extend(batch_lengths)
            if on_batch is not None:
                on_batch(len(batch_lengths))
    return lengths

def _bounded_process_map(executor, function, items, max_pending):
    """Yield ordered results while keeping only a small number of tasks in flight."""
    iterator = iter(items)
    pending = deque()
    for _ in range(max_pending):
        try:
            pending.append(executor.submit(function, next(iterator)))
        except StopIteration:
            break
    while pending:
        future = pending.popleft()
        yield future.result()
        try:
            pending.append(executor.submit(function, next(iterator)))
        except StopIteration:
            continue

def filter_row_chunks(chunks, tokenizer, options, num_workers: int = 0, with_lengths=False):
    """Yield the accepted rows of each chunk, **in input order**.

    One driver for both callers of :func:`filter_rows_batch`: the dataset loader and
    ``src/data/cache.py``. Serial for ``num_workers <= 1``, otherwise a bounded process pool
    running the *same* chunk function, so the two modes cannot drift apart. Order is preserved
    (the pool map is bounded and drained in submission order), which is what keeps a parallel
    cache build byte-identical to a serial one.

    With ``with_lengths`` each item is ``(accepted_rows, context_lengths)`` instead of just the
    rows; the lengths come from the rule check the filter already ran, so they cost nothing.
    """
    if num_workers > 1:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_filter_worker,
            initargs=(tokenizer, options, with_lengths),
        ) as executor:
            yield from _bounded_process_map(
                executor, _filter_row_batch_in_worker, chunks, num_workers * 2
            )
    else:
        for chunk in chunks:
            if with_lengths:
                lengths: list[int] = []
                yield filter_rows_batch(chunk, tokenizer, options, lengths), lengths
            else:
                yield filter_rows_batch(chunk, tokenizer, options)

def progress_enabled() -> bool:
    """Only the main process draws progress bars; one shared rank check, not a second one."""
    return is_main_process()
