from __future__ import annotations


def ddp_kwargs_for_stage(train_stage: str):
    if train_stage != "two_stage":
        return None
    from accelerate import DistributedDataParallelKwargs

    return [DistributedDataParallelKwargs(find_unused_parameters=True)]
