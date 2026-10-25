from __future__ import annotations


def resume_plan(step: int, steps_per_epoch: int, epochs: int) -> tuple[int, int]:
    """Where a resumed run picks up: ``(epoch to start at, batches to skip inside it)``.

    A checkpoint stores the optimizer/scheduler/step but not the position in the data
    stream, so before this a resumed run re-consumed the epoch from batch 0 *and* ran
    another full ``epochs`` pass on top of the loaded step, blowing past ``total_steps``.
    The sampler is deterministic for a fixed rank count, so the batches the run already
    saw can simply be skipped and the step budget stays authoritative.
    """
    if steps_per_epoch <= 0:
        return epochs, 0
    epoch = min(step // steps_per_epoch, epochs)
    if epoch >= epochs:
        return epochs, 0
    return epoch, step - epoch * steps_per_epoch


def optimizer_steps(micro_batches: int, accumulation: int) -> int:
    """Optimizer steps one epoch of ``micro_batches`` micro-batches produces.

    With accumulation the loop still consumes every batch, but only every ``accumulation``-th
    one closes an optimizer step (the scheduler, the evaluation cadence, the checkpoint step
    and ``total_steps`` all count optimizer steps).
    """
    if micro_batches <= 0:
        return 0
    return -(-int(micro_batches) // max(1, int(accumulation)))


def micro_batch_skip(step: int, steps_per_epoch: int, accumulation: int) -> int:
    """Micro-batches of the resume epoch that a run starting at optimizer ``step`` already saw.

    ``resume_plan`` reports the position in *optimizer* steps while the loop counts
    micro-batches; a checkpoint is only ever written on a group boundary, so the product is
    exact and the resumed run continues in the middle of no group.
    """
    if steps_per_epoch <= 0:
        return 0
    return (step % steps_per_epoch) * max(1, int(accumulation))


