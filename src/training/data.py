from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from src.cache_build import DEFAULT_BUILD_WORKERS as BUILD_WORKERS
from src.cache_build import ensure_cache
from src.data import SortishSampler
from src.evaluator import Evaluator
from src.pipeline import (
    data_cache_dir,
    evaluation_window_loader,
    make_collate,
    make_context_dataset,
    resolve_split_limit,
)
from src.training.helpers import _process_index, _world_size


def prepare_training_data(cfg, tokenizer, model, accelerator, device, resume, configure_run_paths, console):
    def print_cache(message):
        if accelerator is None or accelerator.is_main_process:
            print(message)

    # The filtered cache (`cache/train.jsonl` + `cache/validation.jsonl`) is built from the
    # raw meta splits with this config's filters. Building it here keeps "which rows does this
    # run see" in one place; the work happens once (file-locked) no matter how many ranks start
    # together, and `scripts/data/build_cache.py` produces the same files by hand.
    if not cfg.data.train_conversation_file:
        built = ensure_cache(
            cfg.data,
            tokenizer,
            splits=("train", "validation"),
            quiet=accelerator is not None and not accelerator.is_main_process,
            # One pool for the whole run: the file lock means only one rank builds.
            num_workers=BUILD_WORKERS,
        )
        if accelerator is not None:
            accelerator.wait_for_everyone()
        if built and accelerator is not None and accelerator.is_main_process:
            console.print(
                f"[bold yellow]Cache rebuilt[/bold yellow]  {', '.join(built)}"
            )

    train_file = Path(cfg.data.split_file("train"))
    validation_source = Path(cfg.data.validation_source())
    for path, what, build_with in (
        (train_file, "training", "train"),
        (validation_source, "validation", "validation"),
    ):
        if not path.is_file():
            hint = f"  python scripts/data/build_cache.py --config <this config> --splits {build_with}"
            if not Path(cfg.data.meta_file(build_with)).is_file():
                hint = f"  python scripts/data/build_1qa.py   # raw meta splits first\n{hint}"
            raise FileNotFoundError(
                f"{what} data not found: {path}\n"
                f"scripts/data/build_cache.py writes it from the raw meta split; run:\n{hint}"
            )
    model.to(device)
    if accelerator is not None:
        accelerator.wait_for_everyone()

    configure_run_paths(cfg, accelerator, resume)
    cache_dir = data_cache_dir(cfg)

    train_dataset = make_context_dataset(
        cfg,
        "train",
        tokenizer,
        limit=resolve_split_limit(cfg, "train"),
        num_workers=cfg.data.train_num_workers,
    )
    train_collate = make_collate(cfg, tokenizer, sample_qa=True, training=True)
    validation_collate = make_collate(cfg, tokenizer, sample_qa=False)
    sampler = SortishSampler(
        train_dataset,
        tokenizer,
        cfg.training.batch_size,
        cfg.data.sortish_bucket_multiplier,
        cfg.training.seed,
        cfg.data.chat_template_enable_thinking,
        cache_dir if cfg.data.cache_sortish_lengths else None,
        # Deliberately no `num_workers` here, unlike the DataLoader below. This pass is one
        # batched tokenizer call over the contexts, which the tokenizer already parallelizes
        # internally: measured on 20k real contexts, 14224 context/s serial versus 4433 with 2
        # forked workers and 9985 with 8 (the batches are the whole context text, so pickling
        # them costs more than the tokenizing). It is cached after the first run either way.
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        sampler=sampler,
        collate_fn=train_collate,
        num_workers=cfg.data.train_num_workers,
        pin_memory=cfg.data.train_pin_memory,
        prefetch_factor=(
            cfg.data.train_prefetch_factor if cfg.data.train_num_workers else None
        ),
        persistent_workers=(
            cfg.data.train_persistent_workers
            and cfg.data.train_stage != "two_stage"
            and cfg.data.train_num_workers > 0
        ),
    )
    validation_dataset = make_context_dataset(
        cfg,
        "validation",
        tokenizer,
        limit=resolve_split_limit(cfg, "validation"),
        allow_empty=True,
        num_workers=cfg.data.eval_num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.evaluation.ctx_batch_size,
        shuffle=False,
        collate_fn=validation_collate,
        num_workers=cfg.data.eval_num_workers,
        pin_memory=cfg.data.eval_pin_memory,
        prefetch_factor=(
            cfg.data.eval_prefetch_factor if cfg.data.eval_num_workers else None
        ),
        persistent_workers=cfg.data.eval_persistent_workers
        and cfg.data.eval_num_workers > 0,
    )
    # QA gets one seeded sample per dataset. Recon reuses the sample's unique contexts.
    # Both loaders are sharded manually, so accelerator.prepare must not touch them.
    loader_options = dict(
        num_workers=cfg.data.eval_num_workers,
        pin_memory=cfg.data.eval_pin_memory,
        prefetch_factor=(
            cfg.data.eval_prefetch_factor if cfg.data.eval_num_workers else None
        ),
        persistent_workers=cfg.data.eval_persistent_workers
        and cfg.data.eval_num_workers > 0,
    )
    validation_row_loader, validation_row_base = evaluation_window_loader(
        validation_dataset,
        validation_collate,
        budget=cfg.evaluation.max_eval_contexts_per_dataset,
        unit="rows",
        max_qa_sample_per_context=cfg.data.max_qa_sample_per_context,
        seed=cfg.training.seed,
        rank=_process_index(accelerator),
        world=_world_size(accelerator),
        batch_size=cfg.evaluation.ctx_batch_size,
        max_new_tokens=cfg.evaluation.max_new_tokens,
        **loader_options,
    )
    recon_row_loader, recon_row_base = evaluation_window_loader(
        validation_dataset,
        validation_collate,
        budget=cfg.evaluation.max_eval_contexts_per_dataset,
        unit="selected_contexts",
        selection=validation_row_loader.evaluation_selection,
        max_qa_sample_per_context=cfg.data.max_qa_sample_per_context,
        rank=_process_index(accelerator),
        world=_world_size(accelerator),
        batch_size=cfg.evaluation.ctx_batch_size,
        max_new_tokens=cfg.evaluation.max_new_tokens,
        **loader_options,
    )
    evaluator = Evaluator(tokenizer, cfg)
    return {
        "train_dataset": train_dataset, "train_collate": train_collate, "validation_collate": validation_collate,
        "sampler": sampler, "train_loader": train_loader, "validation_dataset": validation_dataset,
        "validation_loader": validation_loader, "validation_row_loader": validation_row_loader,
        "validation_row_base": validation_row_base, "recon_row_loader": recon_row_loader,
        "recon_row_base": recon_row_base, "evaluator": evaluator,
    }
