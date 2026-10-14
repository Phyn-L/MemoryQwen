from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.config import TrainConfig, dtype_from_name
from utils.machines import machine_names, resolve_aim_repo_path
from src.data import SortishSampler
from src.evaluator import Evaluator
from src.losses import objective_total, scheduled_task_weights, supervision_objectives
from src.model import load_model
from src.metrics import METRIC_KEYS
from src.pipeline import (
    make_collate,
    make_context_dataset,
    data_cache_dir,
    new_run_name,
    resolve_split_limit,
)
from src.data_sampling import sample_validation_by_dataset
from utils import CheckpointManager, build_optimizer, build_scheduler, seed_everything

AUTOREGRESSIVE_SECTION = "val_autoregressive"
ANSWER_METRIC_KEYS = (*METRIC_KEYS, "first_token_em", "accuracy")


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


def should_evaluate(step: int, every: int, total_steps: int) -> bool:
    """Whether an evaluation runs at ``step``.

    The cadences are modulo-based, so a cadence that does not divide the run -- every 1000
    steps over a 7890-step single epoch, say -- would leave the last reported number 10%
    short of the end of training, and that final number is the one a short run is read
    for. The last step is therefore always evaluated, whatever the cadence says.
    """
    return step % every == 0 or step >= total_steps


def _gradient_norm(parameters) -> torch.Tensor:
    norms = [
        torch.linalg.vector_norm(parameter.grad.detach().float())
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack(norms))


def eval_log_payloads(task_metrics=None):
    """Log generated-answer metrics, with a separate group per task embedding."""
    if task_metrics is None:
        return []
    payload = {}
    for task_name, metrics in task_metrics.items():
        section = f"{AUTOREGRESSIVE_SECTION}_{task_name}"
        for key, value in metrics.items():
            if key in ANSWER_METRIC_KEYS:
                payload[f"{section}/{key}"] = value
            elif key.startswith("dataset/") and key.rsplit("/", 1)[-1] in {
                *METRIC_KEYS,
                "count",
            }:
                dataset, metric = key.removeprefix("dataset/").rsplit("/", 1)
                dataset = dataset.replace("/", "_").replace("(", "_").replace(")", "")
                metric = {"em": "EM", "f1": "F1", "rouge_l": "ROUGE-L"}.get(
                    metric, metric
                )
                payload[f"{section}/{dataset}_{metric}"] = value
    return [payload] if payload else []


def format_eval_metrics_table(task_metrics) -> str:
    """Render autoregressive evaluation metrics as a compact terminal table."""
    if not task_metrics:
        return ""
    lines = []
    for task_name, metrics in task_metrics.items():
        dataset_names = sorted(
            {
                key.removeprefix("dataset/").rsplit("/", 1)[0]
                for key in metrics
                if key.startswith("dataset/") and key.endswith("/count")
            }
        )
        rows = []
        for dataset in dataset_names:
            rows.append(
                [
                    dataset,
                    *(f"{metrics.get(f'dataset/{dataset}/{key}', 0.0):.4f}"
                      for key in ("em", "f1", "rouge_l")),
                    "-",
                    "-",
                    f"{metrics.get(f'dataset/{dataset}/count', 0.0):.0f}",
                ]
            )
        rows.append(
            [
                "overall",
                *(f"{metrics.get(key, 0.0):.4f}" for key in ("em", "f1", "rouge_l")),
                f"{metrics['first_token_em']:.4f}" if "first_token_em" in metrics else "-",
                f"{metrics['accuracy']:.4f}" if "accuracy" in metrics else "-",
                f"{sum(float(row[-1]) for row in rows):.0f}",
            ]
        )
        headers = ["dataset", "EM", "F1", "ROUGE-L", "first_token_em", "accuracy", "count"]
        widths = [
            max(len(headers[index]), *(len(row[index]) for row in rows))
            for index in range(len(headers))
        ]
        separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
        lines.append(f"[eval] {task_name}")
        lines.append(separator)
        lines.append("| " + " | ".join(
            header.ljust(widths[index]) for index, header in enumerate(headers)
        ) + " |")
        lines.append(separator)
        for row in rows:
            lines.append("| " + " | ".join(
                value.rjust(widths[index]) if index else value.ljust(widths[index])
                for index, value in enumerate(row)
            ) + " |")
        lines.append(separator)
    return "\n".join(lines)


def _test_ids(path: Path) -> set[tuple[str, str]]:
    result = set()
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    result.add((str(row.get("dataset", "")), str(row.get("id", ""))))
    return result


def _configure_run_paths(cfg: TrainConfig, accelerator, resume: str | None) -> None:
    if resume:
        run_dir = Path(resume).resolve().parent
        cfg.checkpoint.output_dir = str(run_dir)
        cfg.logging.aim_run_name = cfg.logging.aim_run_name or run_dir.name
        return
    is_main = accelerator is None or accelerator.is_main_process
    run_name = (
        (cfg.logging.aim_run_name or new_run_name(cfg.model.name_or_path))
        if is_main
        else None
    )
    num_processes = accelerator.num_processes if accelerator is not None else 1
    if num_processes > 1:
        names = [run_name]
        torch.distributed.broadcast_object_list(names, src=0)
        run_name = names[0]
    cfg.logging.aim_run_name = run_name
    # Both explicit and generated names live under the configured output root.
    cfg.checkpoint.output_dir = str(Path(cfg.checkpoint.output_dir) / run_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_baseline.yaml")
    parser.add_argument(
        "--machine",
        choices=machine_names(),
        default=None,
        help="Take MODEL_ROOT / DATA_ROOT from "
        "utils/machines.py::MACHINES. Overrides the config's `machine:` field; can also "
        "come from the MACHINE environment variable or, failing that, the hostname.",
    )
    parser.add_argument("--resume")
    parser.add_argument(
        "--init-from",
        help="Initialize model weights only; start a new optimizer, schedule and run.",
    )
    parser.add_argument("--aim-run-id")
    args = parser.parse_args()

    cfg = TrainConfig.from_file(args.config, machine=args.machine)
    if args.init_from:
        cfg.checkpoint.init_from = str(Path(args.init_from).resolve())
    if args.resume and cfg.checkpoint.init_from:
        parser.error(
            "--resume and checkpoint.init_from/--init-from are mutually exclusive"
        )
    cfg.validate()
    seed_everything(cfg.training.seed, deterministic=cfg.training.deterministic)
    tokenizer, model = load_model(cfg)

    accelerator = None
    try:
        from accelerate import Accelerator

        configured_dtype = dtype_from_name(cfg.model.torch_dtype)
        mixed_precision = {
            torch.float32: "no",
            torch.float16: "fp16",
            torch.bfloat16: "bf16",
        }[configured_dtype]
        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision=mixed_precision if torch.cuda.is_available() else "no",
        )
        device = accelerator.device
    except ImportError:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise RuntimeError("Multi-GPU training requires accelerate")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(__file__).resolve().parents[1] / "contexts" / "1qa"
    validation_source = Path(
        cfg.data.validation_source_file or data_dir / "1qa-validation.jsonl"
    )
    validation_dir = Path(cfg.data.validation_output_dir or data_dir / "1qa-val")
    sampled_validation = validation_dir / "1qa-val.jsonl"
    validation_counts = None
    datasets = tuple(cfg.data.validation_datasets or ())
    default_per_dataset = (cfg.data.validation_max_samples or 1280) // max(
        len(datasets) or 5, 1
    )
    sample_limits = cfg.data.validation_max_samples_per_dataset or {
        dataset: default_per_dataset for dataset in datasets
    }
    if not sample_limits:
        sample_limits = {
            "squad": 256,
            "msmarcov1.1": 256,
            "msmarcov2.1": 256,
            "hotpotqa(distractor)": 256,
            "race": 256,
        }
    validation_context_limit = cfg.data.validation_max_context_tokens
    if validation_context_limit:
        cfg.data.max_context_tokens = validation_context_limit
    test_ids = _test_ids(data_dir / "1qa-test.jsonl")
    if accelerator is None or accelerator.is_main_process:
        sampled_validation, validation_counts = sample_validation_by_dataset(
            validation_source,
            validation_dir,
            samples_per_dataset=sample_limits,
            seed=cfg.training.seed,
            excluded_ids=test_ids,
            tokenizer=tokenizer,
            data_config=cfg.data,
            datasets=datasets or None,
            num_workers=cfg.data.validation_sampling_workers,
        )
    if accelerator is not None:
        accelerator.wait_for_everyone()
    cfg.data.train_conversation_file = str(data_dir / "1qa-train.jsonl")
    cfg.data.validation_conversation_file = str(sampled_validation)
    cfg.data.conversation_file = cfg.data.train_conversation_file
    model.to(device)
    if accelerator is not None:
        accelerator.wait_for_everyone()

    _configure_run_paths(cfg, accelerator, args.resume)
    cache_dir = data_cache_dir()

    train_dataset = make_context_dataset(
        cfg,
        "train",
        tokenizer,
        limit=resolve_split_limit(cfg, "train"),
        num_workers=cfg.data.num_workers,
    )
    train_collate = make_collate(cfg, tokenizer, sample_qa=True, training=True)
    sampler = SortishSampler(
        train_dataset,
        tokenizer,
        cfg.training.batch_size,
        cfg.data.sortish_bucket_multiplier,
        cfg.training.seed,
        cfg.data.chat_template_enable_thinking,
        cache_dir if cfg.data.cache_sortish_lengths else None,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        sampler=sampler,
        collate_fn=train_collate,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers and cfg.data.num_workers > 0,
    )
    validation_dataset = make_context_dataset(
        cfg,
        "validation",
        tokenizer,
        limit=resolve_split_limit(cfg, "validation"),
        allow_empty=True,
        num_workers=cfg.data.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=make_collate(cfg, tokenizer, sample_qa=False),
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers and cfg.data.num_workers > 0,
    )
    # A second, deliberately *unprepared* loader over the same dataset. The evaluation
    # budget (`evaluation.evaluation_max_samples`) is global, and the evaluator slices it into
    # contiguous windows of the global row order; `accelerator.prepare` shards by batch, so
    # the prepared loader cannot express "rows 1024..2047 of the split". Skipping the
    # batches outside a window happens before any compute, so this costs iteration only.
    validation_row_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=make_collate(cfg, tokenizer, sample_qa=False),
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers and cfg.data.num_workers > 0,
    )
    evaluator = Evaluator(tokenizer, cfg)
    effective_loader_len = len(train_loader)
    if accelerator is not None and accelerator.num_processes > 1:
        effective_loader_len = (
            effective_loader_len + accelerator.num_processes - 1
        ) // accelerator.num_processes
    # One optimizer step per DataLoader batch: gradient accumulation is not used.
    total_steps = max(1, effective_loader_len * cfg.training.epochs)
    optimizer = build_optimizer(model, cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg.scheduler, total_steps)
    # Echo the resolved schedule once. On a single-epoch run every cadence and the warm-up
    # are read against the step budget, and a batch/process mismatch changes that budget
    # silently: this line is the number the progress bar will show, printed before any
    # compute is spent.
    if accelerator is None or accelerator.is_main_process:
        ranks = 1 if accelerator is None else accelerator.num_processes
        print(
            "schedule: "
            f"machine={cfg.machine or 'config-defaults'} "
            f"steps={total_steps} (epochs={cfg.training.epochs}) "
            f"batch={cfg.training.batch_size}x{ranks} ranks={cfg.training.batch_size * ranks} "
            f"warmup={cfg.scheduler.warmup_steps} "
            f"autoregressive_every={cfg.evaluation.autoregressive_every} "
            f"checkpoint_every={cfg.evaluation.autoregressive_every} "
            f"log_every={cfg.logging.log_every} "
            f"output_dir={cfg.checkpoint.output_dir}",
            flush=True,
        )
    manager = CheckpointManager(cfg.checkpoint.output_dir)
    if cfg.checkpoint.init_from and not args.resume:
        # Use a separate manager so phase-one selection metadata and RNG do not leak.
        CheckpointManager(cfg.checkpoint.output_dir).load(
            cfg.checkpoint.init_from,
            model,
            restore_rng=False,
        )

    step = 0
    if args.resume:
        step, saved_run_id = manager.load(args.resume, model, optimizer, scheduler)
        if not cfg.logging.aim_run_id:
            cfg.logging.aim_run_id = saved_run_id
        # The sampler is deterministic only for a fixed sharding; a different rank count
        # would make "skip the batches this run already saw" skip the wrong ones.
        ranks = 1 if accelerator is None else accelerator.num_processes
        saved_ranks = manager.loaded.get("world_size", ranks)
        if saved_ranks != ranks:
            print(
                f"resume: WARNING checkpoint was written with {saved_ranks} ranks but this run "
                f"uses {ranks}; the batch sharding differs, so the skipped batches are not the "
                f"ones this run saw.",
                flush=True,
            )
    start_epoch, skip_batches = resume_plan(
        step, effective_loader_len, cfg.training.epochs
    )
    if step:
        print(
            f"resume: step {step}/{total_steps} -> epoch {start_epoch + 1}/{cfg.training.epochs}, "
            f"skipping {skip_batches} already-consumed batches",
            flush=True,
        )
    finished = False
    if args.aim_run_id:
        cfg.logging.aim_run_id = args.aim_run_id

    if accelerator is not None:
        # The scheduler is deliberately NOT prepared. `accelerator.prepare` wraps an
        # LRScheduler in `AcceleratedScheduler`, whose `step()` advances the wrapped
        # scheduler `num_processes` times per call when ``split_batches`` is False (the
        # default): with 2 ranks a cosine built for 8 steps is finished after 4 loop
        # steps, and 8 ranks consume the whole schedule in the first eighth of the run.
        # HF's cosine keeps being evaluated past its end, so the LR then *oscillates*
        # between zero and the peak instead of annealing. Every rank steps its own
        # unwrapped scheduler once per optimizer step, which is the intended schedule and
        # keeps the ranks identical.
        model, optimizer, train_loader, validation_loader = accelerator.prepare(
            model,
            optimizer,
            train_loader,
            validation_loader,
        )

    run = None
    if accelerator is None or accelerator.is_main_process:
        try:
            from aim import Run

            # Launch override, explicit config, selected machine, then output root.
            aim_repo = resolve_aim_repo_path(
                cfg.machine, cfg.logging.aim_repo_path, cfg.checkpoint.output_dir
            )
            run_kwargs = {
                "repo": aim_repo,
                "experiment": cfg.logging.aim_project,
            }
            if cfg.logging.aim_run_id:
                run_kwargs["run_hash"] = cfg.logging.aim_run_id
            run = Run(**run_kwargs)
            run["name"] = cfg.logging.aim_run_name
            run["hparams"] = cfg.to_dict()
        except ImportError:
            pass

    model.train()
    progress = tqdm(
        total=total_steps,
        desc="Training",
        unit="step",
        dynamic_ncols=True,
        disable=not (accelerator is None or accelerator.is_main_process),
        initial=step,
    )
    for epoch in range(start_epoch, cfg.training.epochs):
        sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < skip_batches:
                continue
            if step >= total_steps:
                finished = True
                break
            ids = {k: v.to(device) for k, v in batch.items() if k != "records"}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_started = time.perf_counter()
            base_model = (
                accelerator.unwrap_model(model) if accelerator is not None else model
            )
            embedding = base_model.qwen.get_input_embeddings()
            recon_batch = {
                "context_indices": ids["recon_context_indices"],
                "context_embeds": embedding(ids["recon_target_ids"]),
                "context_mask": ids["recon_target_mask"],
                "labels": ids["recon_labels"],
            }
            output = model(
                embedding(ids["context_ids"]),
                ids["context_mask"],
                embedding(ids["qa_input_ids"]),
                ids["qa_input_mask"],
                embedding(ids["qa_target_ids"]),
                ids["qa_target_mask"],
                ids["qa_labels"],
                ids["qa_context_indices"],
                context_ids=ids["context_ids"],
                objective_config=cfg.memory,
                loss_only=True,
                recon_batch=recon_batch,
            )
            task_ids = torch.cat(
                [
                    torch.ones(
                        ids["qa_input_ids"].size(0), dtype=torch.long, device=device
                    ),
                    torch.zeros(
                        ids["recon_target_ids"].size(0), dtype=torch.long, device=device
                    ),
                ]
            )
            qa_weight, recon_weight, effective_ratio = scheduled_task_weights(
                step, total_steps, cfg.data
            )
            supervised, task_terms = supervision_objectives(
                output.supervision_losses,
                task_ids,
                cfg.memory,
                qa_weight=qa_weight * cfg.memory.qa_weight,
                recon_weight=recon_weight * cfg.memory.recon_weight,
            )
            auxiliary_terms = output.objective_terms or {}
            total = supervised + objective_total(
                supervised * 0, auxiliary_terms, cfg.memory
            )
            terms = {
                **auxiliary_terms,
                **task_terms,
                "loss": total,
                "qa_weight": total.detach() * 0 + qa_weight * cfg.memory.qa_weight,
                "recon_weight": total.detach() * 0
                + recon_weight * cfg.memory.recon_weight,
                "recon_ratio": total.detach() * 0 + effective_ratio,
            }
            if accelerator is not None:
                accelerator.backward(total)
            else:
                total.backward()
            named_parameters = list(base_model.named_parameters())
            resampler_grad_norm = _gradient_norm(
                parameter
                for name, parameter in named_parameters
                if name.startswith("context_resampler.") and "layer_gates" not in name
            )
            metalora_grad_norm = _gradient_norm(
                parameter
                for name, parameter in named_parameters
                if "lora_A" in name or "lora_B" in name
            )
            layer_gates_grad_norm = _gradient_norm(
                parameter
                for name, parameter in named_parameters
                if name == "context_resampler.layer_gates"
            )
            recon_task_embedding_grad_norm = (
                base_model.recon_task_embedding.grad.detach().float().norm()
                if base_model.recon_task_embedding.grad is not None
                else torch.zeros((), device=device)
            )
            step_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.training.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_seconds = max(time.perf_counter() - step_started, 1e-9)
            step += 1
            progress.update(1)
            progress.set_postfix(
                loss=f"{float(total.detach()):.4f}",
                ppl=f"{float(torch.exp(terms['qa_loss'].detach().float().clamp(max=20))):.4f}",
            )

            if run and step % cfg.logging.log_every == 0:
                # `train/lr` is logged next to the losses on purpose: the LR schedule is
                # the one training-side scalar that silently went wrong before (accelerate
                # stepping it once per rank, see the prepare() comment), and without this
                # key a distorted schedule is invisible in the run history.
                payload = {
                    "loss/total": float(terms["loss"].detach()),
                    "loss/qa": float(terms["qa_loss"].detach()),
                    "loss/recon": float(terms["recon_loss"].detach()),
                    "loss/distill": float(terms["distill_loss"].detach()),
                    "train/lr": float(scheduler.get_last_lr()[0]),
                    "loss/ppl": float(
                        torch.exp(terms["qa_loss"].detach().float().clamp(max=20))
                    ),
                    "loss/qa_ppl": float(
                        torch.exp(terms["qa_loss"].detach().float().clamp(max=20))
                    ),
                    "loss/recon_ppl": float(
                        torch.exp(terms["recon_loss"].detach().float().clamp(max=20))
                    ),
                    "grad_norm/total": float(step_grad_norm.detach()),
                    "grad_norm/resampler": float(resampler_grad_norm),
                    "grad_norm/metalora": float(metalora_grad_norm),
                    "grad_norm/layer_gates": float(layer_gates_grad_norm),
                    "grad_norm/recon_task_embedding": float(recon_task_embedding_grad_norm),
                    "data/qa_probability": float(1.0 - effective_ratio),
                    "data/recon_probability": float(effective_ratio),
                }
                for name, value in payload.items():
                    run.track(value, name=name, step=step)
            is_main = accelerator is None or accelerator.is_main_process
            # Validation runs on EVERY rank.  accelerator.prepare shards the validation
            # loader, so evaluating only on the main process would (a) score one shard
            # instead of the whole split and (b) leave the other ranks waiting in the next
            # DDP collective for the entire evaluation, which the NCCL watchdog aborts
            # after 10 minutes ("Watchdog caught collective operation timeout").  The
            # evaluators all-reduce their accumulators, so every rank must call them.
            metrics = None
            if len(validation_loader) and should_evaluate(
                step, cfg.evaluation.autoregressive_every, total_steps
            ):
                if is_main:
                    print(
                        f"\n[eval] step={step}: starting QA autoregressive validation "
                        f"(max_samples={cfg.evaluation.evaluation_max_samples}, "
                        f"max_new_tokens={cfg.evaluation.max_new_tokens})",
                        flush=True,
                    )
                metrics = evaluator.compare_reader_tasks(
                    base_model,
                    validation_loader,
                    device,
                    max_qa=cfg.evaluation.evaluation_max_samples,
                    row_loader=validation_row_loader,
                    include_recon=False,
                )
                if is_main:
                    print(
                        f"\n[eval] step={step}: validation complete\n"
                        f"{format_eval_metrics_table(metrics)}",
                        flush=True,
                    )
            if run and is_main:
                for payload in eval_log_payloads(metrics):
                    for name, value in payload.items():
                        run.track(value, name=name, step=step)
            if accelerator is not None:
                # Resynchronise explicitly so a rank whose shard finished early cannot
                # race ahead into the next training step.
                accelerator.wait_for_everyone()
            if (metrics is not None) and torch.cuda.is_available():
                # An evaluation allocates a very different tensor mix from a training step
                # (whole contexts through every decoder, a full validation loader), and the
                # allocator keeps those blocks cached afterwards. On the H200 that residue
                # was enough to break the *next* step's backward with "5.31 GiB reserved but
                # unallocated" while a 3.40 GiB block was requested -- the ON arm died right
                # after its first evaluation at step 500. Handing the cache back to the
                # driver costs a fraction of a second once per evaluation interval, and only
                # on the steps that actually evaluated.
                torch.cuda.empty_cache()
            if metrics is not None and is_main:
                checkpoint_model = (
                    accelerator.unwrap_model(model)
                    if accelerator is not None
                    else model
                )
                manager.save(
                    checkpoint_model,
                    optimizer,
                    scheduler,
                    step,
                    cfg.to_dict(),
                    validation_f1=metrics["qa_task"]["f1"],
                    aim_run_id=getattr(run, "hash", None),
                    world_size=1 if accelerator is None else accelerator.num_processes,
                )
        if finished:
            break

    if accelerator is None or accelerator.is_main_process:
        checkpoint_model = (
            accelerator.unwrap_model(model) if accelerator is not None else model
        )
        manager.save(
            checkpoint_model,
            optimizer,
            scheduler,
            step,
            cfg.to_dict(),
            final=True,
            aim_run_id=getattr(run, "hash", None),
            world_size=1 if accelerator is None else accelerator.num_processes,
        )
    progress.close()
    if run:
        run.close()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
