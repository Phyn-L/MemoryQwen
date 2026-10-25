from __future__ import annotations

import math


def should_evaluate(step: int, every: int, total_steps: int) -> bool:
    """Whether an evaluation runs at ``step``.

    The cadences are modulo-based, so a cadence that does not divide the run -- every 1000
    steps over a 7890-step single epoch, say -- would leave the last reported number 10%
    short of the end of training, and that final number is the one a short run is read
    for. The last step is therefore always evaluated, whatever the cadence says.
    """
    return step % every == 0 or step >= total_steps


def step_postfix(
    terms: dict[str, float], micro_batches: int, max_ppl_exponent: float = 20.0
):
    """The tqdm fields for the accumulation group that is currently running.

    Values are sums over the group, so the display is the step's *mean* rather than the last
    micro-batch. It is one dict on purpose: ``set_postfix`` **replaces** the displayed fields,
    and a second call carrying only ``loss`` used to wipe the perplexity immediately after every
    optimizer step (with the default accumulation of 1, that was every step). It is returned as a
    plain dict in display order and passed positionally, because tqdm sorts keyword arguments.
    """
    means = {name: value / max(micro_batches, 1) for name, value in terms.items()}
    postfix = {"loss": f"{means.get('loss', 0.0):.4f}"}
    if "qa_loss" in means:
        postfix["qa"] = f"{means['qa_loss']:.4f}"
    if "recon_loss" in means:
        postfix["recon"] = f"{means['recon_loss']:.4f}"
    if "qa_loss" in means:
        # Same definition as `loss/ppl` in the SwanLab payload: perplexity of the QA objective.
        postfix["ppl"] = f"{math.exp(min(means['qa_loss'], max_ppl_exponent)):.2f}"
    return postfix


