from __future__ import annotations

import argparse
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.config import TrainConfig, dtype_from_name
from utils.machines import machine_names
from src.data import SortishSampler
from src.evaluator import Evaluator
from src.losses import (
    combine_losses,
    context_lm_loss,
    kl_distill_loss,
    qa_loss,
    reconstruction_loss,
    sample_positions,
    sequence_lm_loss,
)
from src.model import load_model
from src.metrics import METRIC_KEYS
from src.pipeline import (
    make_collate,
    make_context_dataset,
    model_cache_dir,
    new_run_name,
    resolve_split_limit,
)
from utils import CheckpointManager, build_optimizer, build_scheduler

# W&B groups panels by the FIRST path component of a logged key, so the two evaluation
# modes must lead the key to become two sections. Logging `val/teacher_forced/...` and
# `val/autoregressive/...` instead puts both under a single crowded `val` section.
TEACHER_FORCED_SECTION = "val_teacher_forced"
AUTOREGRESSIVE_SECTION = "val_autoregressive"
# Answer-quality metrics, reported with identical keys by both modes so the two sections
# line up panel for panel.
ANSWER_METRIC_KEYS = (*METRIC_KEYS, "first_token_em")
# Loss-like scalars only exist on the teacher-forced pass (the autoregressive pass can run
# one internally to obtain them, but they are still that pass's numbers).
LOSS_KEYS = ("loss", "qa_loss", "ppl", "reconstruction_loss", "ae_loss", "distill_loss")


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


def eval_log_payloads(teacher_metrics=None, autoregressive_metrics=None):
    """Group evaluation scalars into their W&B sections.

    Two sections, each carrying the same answer-quality keys, so the two evaluation modes
    line up panel for panel. Loss-like scalars appear in the teacher-forced section only:
    when `teacher_metrics` is None the autoregressive pass computed them internally, but
    they are still that pass's numbers and must not be duplicated under the other section.

    Training-side losses (including ``ae_loss``/``distill_loss``) are logged separately as
    ``train/*`` every ``logging.log_every`` steps by the training loop, so they must not be
    folded in here.
    """
    payloads = []
    if teacher_metrics is not None:
        payloads.append(
            {
                f"{TEACHER_FORCED_SECTION}/{key}": value
                for key, value in teacher_metrics.items()
            }
        )
    if autoregressive_metrics is not None:
        payloads.append(
            {
                f"{AUTOREGRESSIVE_SECTION}/{key}": autoregressive_metrics[key]
                for key in ANSWER_METRIC_KEYS
                if key in autoregressive_metrics
            }
        )
        if teacher_metrics is None:
            loss_only = {
                key: value
                for key, value in autoregressive_metrics.items()
                if key in LOSS_KEYS
            }
            if loss_only:
                payloads.append(
                    {
                        f"{TEACHER_FORCED_SECTION}/{key}": value
                        for key, value in loss_only.items()
                    }
                )
    return payloads


def _configure_run_paths(cfg: TrainConfig, accelerator, resume: str | None) -> None:
    if resume:
        run_dir = Path(resume).resolve().parent
        cfg.checkpoint.output_dir = str(run_dir)
        cfg.logging.wandb_run_name = cfg.logging.wandb_run_name or run_dir.name
        return
    is_main = accelerator is None or accelerator.is_main_process
    run_name = new_run_name(cfg.model.name_or_path) if is_main else None
    num_processes = accelerator.num_processes if accelerator is not None else 1
    if num_processes > 1:
        names = [run_name]
        torch.distributed.broadcast_object_list(names, src=0)
        run_name = names[0]
    cfg.logging.wandb_run_name = run_name
    # The run keeps its own directory (a re-run must not clobber the previous one), but it
    # lives *under* the config's ``checkpoint.output_dir`` instead of replacing it. Before
    # this, every run went to ``outputs/<auto name>`` and the configured path was silently
    # ignored -- which is why a crashed A/B arm's checkpoints could not be found afterwards,
    # let alone resumed. Now an arm's checkpoints are all under ``outputs/ab_h200_on/``.
    cfg.checkpoint.output_dir = str(Path(cfg.checkpoint.output_dir) / run_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/4090/qwen-1.7b/baseline/train_baseline.yaml")
    parser.add_argument(
        "--machine",
        choices=machine_names(),
        default=None,
        help="Take MODEL_ROOT / DATA_ROOT (and the machine's wandb mode) from "
             "utils/machines.py::MACHINES. Overrides the config's `machine:` field; can also "
             "come from the MACHINE environment variable or, failing that, the hostname.",
    )
    parser.add_argument("--resume")
    parser.add_argument("--wandb-run-id")
    args = parser.parse_args()

    cfg = TrainConfig.from_file(args.config, machine=args.machine)
    cfg.validate()
    random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    _configure_run_paths(cfg, accelerator, args.resume)
    cache_dir = model_cache_dir(cfg.model.name_or_path)

    train_collate = make_collate(cfg, tokenizer, sample_qa=True)
    train_dataset = make_context_dataset(
        cfg,
        "train",
        tokenizer,
        limit=resolve_split_limit(cfg, "train"),
    )
    sampler = SortishSampler(
        train_dataset,
        tokenizer,
        cfg.training.batch_size,
        cfg.data.sortish_bucket_multiplier,
        cfg.training.seed,
        cfg.data.use_chat_template,
        cfg.data.chat_template_enable_thinking,
        cache_dir if cfg.data.cache_sortish_lengths else None,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        sampler=sampler,
        collate_fn=train_collate,
    )
    validation_dataset = make_context_dataset(
        cfg,
        "validation",
        tokenizer,
        limit=resolve_split_limit(cfg, "validation"),
        allow_empty=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=make_collate(cfg, tokenizer, sample_qa=False),
    )
    # A second, deliberately *unprepared* loader over the same dataset. The autoregressive
    # budget (`evaluation.autoregressive_max_qa`) is global, and the evaluator slices it into
    # contiguous windows of the global row order; `accelerator.prepare` shards by batch, so
    # the prepared loader cannot express "rows 1024..2047 of the split". Skipping the
    # batches outside a window happens before any compute, so this costs iteration only.
    validation_row_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=make_collate(cfg, tokenizer, sample_qa=False),
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
            f"teacher_forced_every={cfg.evaluation.teacher_forced_every} "
            f"autoregressive_every={cfg.evaluation.autoregressive_every} "
            f"checkpoint_every={cfg.evaluation.autoregressive_every} "
            f"log_every={cfg.logging.log_every} "
            f"output_dir={cfg.checkpoint.output_dir}",
            flush=True,
        )
    manager = CheckpointManager(cfg.checkpoint.output_dir)

    step = 0
    if args.resume:
        step, saved_run_id = manager.load(args.resume, model, optimizer, scheduler)
        if not cfg.logging.wandb_run_id:
            cfg.logging.wandb_run_id = saved_run_id
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
    start_epoch, skip_batches = resume_plan(step, effective_loader_len, cfg.training.epochs)
    if step:
        print(
            f"resume: step {step}/{total_steps} -> epoch {start_epoch + 1}/{cfg.training.epochs}, "
            f"skipping {skip_batches} already-consumed batches",
            flush=True,
        )
    finished = False
    if args.wandb_run_id:
        cfg.logging.wandb_run_id = args.wandb_run_id

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
        model, optimizer, train_loader, validation_loader = (
            accelerator.prepare(
                model,
                optimizer,
                train_loader,
                validation_loader,
            )
        )

    run = None
    if accelerator is None or accelerator.is_main_process:
        try:
            import wandb

            run = wandb.init(
                project=cfg.logging.wandb_project,
                name=cfg.logging.wandb_run_name,
                id=cfg.logging.wandb_run_id,
                resume="allow" if cfg.logging.wandb_run_id else None,
                mode=cfg.logging.wandb_mode,
                config=cfg.to_dict(),
            )
        except ImportError:
            pass

    model.train()
    progress = tqdm(
        total=total_steps,
        desc="Training",
        unit="step",
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
            base_model = (
                accelerator.unwrap_model(model) if accelerator is not None else model
            )
            embedding = base_model.qwen.get_input_embeddings()
            output = model(
                embedding(ids["context_ids"]),
                ids["context_mask"],
                embedding(ids["question_ids"]),
                ids["question_mask"],
                embedding(ids["answer_ids"]),
                ids["answer_mask"],
                ids["labels"],
                ids["qa_context_indices"],
                context_ids=ids["context_ids"],
                context_lm_positions=cfg.memory.context_lm_positions,
            )
            qa = qa_loss(output.logits, output.labels)
            if cfg.memory.reconstruction_loss == "context_lm":
                # Context next-token prediction through the memory, in nats/token.
                # Starts near ln(vocab_size) ~= 11.9, so reconstruction_weight may
                # want to be smaller than for the embedding regression (which sits
                # near 0.08) to keep the two terms on comparable gradient scales.
                reconstruction = context_lm_loss(
                    output.context_lm_hidden,
                    output.context_lm_labels,
                    base_model.context_lm_head,
                    output.context_lm_mask,
                )
            else:
                reconstruction = reconstruction_loss(
                    output.reconstruction,
                    output.context_target,
                    output.context_mask,
                    cfg.memory.reconstruction_cosine_weight,
                    cfg.memory.reconstruction_loss,
                )
            # Memory-prefixed autoencoding: reconstruct the context through the frozen
            # backbone, scored by its own (tied) unembedding. This is the objective the
            # compression literature uses; it starts near the LM's own nats/token instead of
            # near ln(vocab), so its weight is comparable to qa_weight.
            ae = None
            if cfg.memory.ae_lm_weight and output.ae_hidden is not None:
                positions = None
                if cfg.memory.ae_lm_positions > 0:
                    positions = sample_positions(output.ae_mask, cfg.memory.ae_lm_positions)[0]
                ae = sequence_lm_loss(
                    output.ae_hidden,
                    output.ae_labels,
                    base_model.ae_head,
                    output.ae_mask,
                    positions=positions,
                )
            # Distribution-level distillation of the full-context model into the
            # memory-conditioned one. The teacher is the encoder pass's own context rows
            # (plain causal LM, no extra forward), so this costs one vocabulary pass per
            # scored position and no additional backbone pass.
            distill = None
            if cfg.memory.distill_weight and output.ae_teacher_hidden is not None:
                positions = None
                if cfg.memory.distill_positions > 0:
                    positions = sample_positions(output.ae_mask, cfg.memory.distill_positions)[0]
                distill = kl_distill_loss(
                    output.ae_hidden,
                    output.ae_teacher_hidden,
                    base_model.ae_head,
                    output.ae_mask,
                    cfg.memory.distill_temperature,
                    positions=positions,
                    topk=cfg.memory.distill_topk,
                    entropy_weight=cfg.memory.distill_entropy_weight,
                )
            total, terms = combine_losses(
                qa,
                reconstruction,
                cfg.memory.qa_weight,
                cfg.memory.reconstruction_weight,
                ae,
                cfg.memory.ae_lm_weight,
                distill,
                cfg.memory.distill_weight,
            )
            if accelerator is not None:
                accelerator.backward(total)
            else:
                total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.training.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            progress.update(1)
            progress.set_postfix(
                loss=f"{float(total.detach()):.4f}",
                qa=f"{float(qa.detach()):.4f}",
                recon=f"{float(reconstruction.detach()):.4f}",
            )

            if run and step % cfg.logging.log_every == 0:
                # `train/lr` is logged next to the losses on purpose: the LR schedule is
                # the one training-side scalar that silently went wrong before (accelerate
                # stepping it once per rank, see the prepare() comment), and without this
                # key a distorted schedule is invisible in the run history.
                run.log(
                    {
                        **{f"train/{k}": float(v.detach()) for k, v in terms.items()},
                        "train/lr": float(scheduler.get_last_lr()[0]),
                    },
                    step=step,
                )
            is_main = accelerator is None or accelerator.is_main_process
            # Validation runs on EVERY rank.  accelerator.prepare shards the validation
            # loader, so evaluating only on the main process would (a) score one shard
            # instead of the whole split and (b) leave the other ranks waiting in the next
            # DDP collective for the entire evaluation, which the NCCL watchdog aborts
            # after 10 minutes ("Watchdog caught collective operation timeout").  The
            # evaluators all-reduce their accumulators, so every rank must call them.
            # Teacher forcing feeds the answer prefix back in, so its em/f1 mostly reward
            # lexical continuation.  Run it as a diagnostic, and reuse its scalars for the
            # autoregressive block below when both happen to fire on the same step.
            teacher_metrics = None
            if len(validation_loader) and should_evaluate(
                step, cfg.evaluation.teacher_forced_every, total_steps
            ):
                teacher_metrics = evaluator.teacher_forced(
                    base_model, validation_loader, device
                )
            metrics = None
            if len(validation_loader) and should_evaluate(
                step, cfg.evaluation.autoregressive_every, total_steps
            ):
                metrics = evaluator.autoregressive(
                    base_model,
                    validation_loader,
                    device,
                    include_teacher_metrics=teacher_metrics is None,
                    max_qa=cfg.evaluation.autoregressive_max_qa,
                    row_loader=validation_row_loader,
                )
            if run and is_main:
                # Both evaluations are logged together; the step is explicit, so the two
                # sections share a step whether or not both ran.
                for payload in eval_log_payloads(teacher_metrics, metrics):
                    run.log(payload, step=step)
            if accelerator is not None:
                # Resynchronise explicitly so a rank whose shard finished early cannot
                # race ahead into the next training step.
                accelerator.wait_for_everyone()
            if (teacher_metrics is not None or metrics is not None) and torch.cuda.is_available():
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
                    validation_f1=metrics["f1"],
                    wandb_run_id=getattr(run, "id", None),
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
            wandb_run_id=getattr(run, "id", None),
            world_size=1 if accelerator is None else accelerator.num_processes,
        )
    progress.close()
    if run:
        run.finish()


if __name__ == "__main__":
    main()
