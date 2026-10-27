from __future__ import annotations

import math

import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src.evaluation.eval_schedule import should_evaluate, step_postfix
from src.losses import scheduled_task_weights, supervision_objectives
from src.training.stages import sample_task_masks, training_stage
from utils.progress import StepSpeedColumn
from utils.reporting import eval_log_payloads, format_evaluation_report

from .helpers import _gradient_norm, accumulation_context

CONSOLE = Console()

def run_training_loop(
    cfg, model, optimizer, scheduler, train_loader, validation_loader, sampler, train_collate,
    evaluator, validation_row_loader, validation_row_base, recon_row_loader, recon_row_base,
    managers, manager, step, start_epoch, skip_micro_batches, total_epochs, total_steps,
    steps_per_epoch, micro_batches_per_rank, accumulation, accelerator, device, run, trainable,
):
    model.train()
    # Scalar sums of the micro-batches in the group that is currently accumulating; the step
    # that closes the group logs and resets them.
    group_terms: dict[str, torch.Tensor] = {}
    group_micro_batches = 0
    progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        StepSpeedColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[green]{task.fields[status]}"),
        console=CONSOLE,
        disable=not (accelerator is None or accelerator.is_main_process),
    )
    progress_task = progress.add_task(
        "Training",
        total=total_steps,
        completed=step,
        status="",
        speed_start_completed=step,
    )
    progress.start()
    for epoch in range(start_epoch, total_epochs):
        stage = training_stage(cfg.data.train_stage, epoch, cfg.training.epochs)
        active_scheduler = scheduler[stage] if isinstance(scheduler, dict) else scheduler
        manager = managers[stage]
        stage_offset = (
            (0 if stage == "recon" else steps_per_epoch * cfg.training.epochs)
            if cfg.data.train_stage == "two_stage"
            else 0
        )
        stage_total_steps = steps_per_epoch * cfg.training.epochs
        sampler.set_epoch(epoch)
        sampler.set_start_batch(0)
        # Keep the deterministic QA-turn sampling of the collate on the same epoch.
        train_collate.set_epoch(epoch)
        epoch_loader = train_loader
        batch_offset = skip_micro_batches if epoch == start_epoch else 0
        if batch_offset:
            if accelerator is not None:
                # This wraps the prepared batch sampler, so skipped indices never reach
                # Dataset.__getitem__ or collate_fn on any rank.
                from accelerate.data_loader import skip_first_batches

                epoch_loader = skip_first_batches(train_loader, skip_micro_batches)
            else:
                sampler.set_start_batch(skip_micro_batches)
                epoch_loader = train_loader
        if hasattr(epoch_loader, "set_epoch"):
            epoch_loader.set_epoch(epoch)
        for batch_index, batch in enumerate(epoch_loader, start=batch_offset):
            if step >= total_steps:
                break
            # Accumulation: `accumulate` defers the DDP gradient all-reduce to the step
            # that closes the group (and tracks `sync_gradients`); without accelerate the
            # loop below does the same arithmetic by hand.
            sync_now = (
                batch_index + 1
            ) % accumulation == 0 or batch_index + 1 >= micro_batches_per_rank
            with accumulation_context(accelerator, model):
                ids = {
                    k: v.to(device, non_blocking=True)
                    for k, v in batch.items()
                    if k != "records"
                }
                base_model = (
                    accelerator.unwrap_model(model)
                    if accelerator is not None
                    else model
                )
                embedding = base_model.qwen.get_input_embeddings()
                is_two_stage = cfg.data.train_stage == "two_stage"
                run_qa = not is_two_stage or stage == "qa"
                run_recon = not is_two_stage or stage == "recon"
                qa_input_ids = ids["qa_input_ids"] if run_qa else ids["qa_input_ids"][:0]
                qa_input_mask = ids["qa_input_mask"] if run_qa else ids["qa_input_mask"][:0]
                qa_target_ids = ids["qa_target_ids"] if run_qa else ids["qa_target_ids"][:0]
                qa_target_mask = ids["qa_target_mask"] if run_qa else ids["qa_target_mask"][:0]
                qa_labels = ids["qa_labels"] if run_qa else ids["qa_labels"][:0]
                qa_context_indices = ids["qa_context_indices"] if run_qa else ids["qa_context_indices"][:0]
                recon_batch = None
                if run_recon:
                    recon_batch = {
                        "context_indices": ids["recon_context_indices"],
                        "context_embeds": embedding(ids["recon_target_ids"]),
                        "context_mask": ids["recon_target_mask"],
                        "labels": ids["recon_labels"],
                    }
                output = model(
                    embedding(ids["context_ids"]),
                    ids["context_mask"],
                    embedding(qa_input_ids), qa_input_mask,
                    embedding(qa_target_ids), qa_target_mask, qa_labels,
                    qa_context_indices,
                    loss_config=cfg.loss,
                    loss_only=True,
                    recon_batch=recon_batch,
                    question_prefix_embeds=embedding(ids["qa_prefix_ids"]) if run_qa else None,
                    question_prefix_mask=ids["qa_prefix_mask"] if run_qa else None,
                    question_readout_mask=ids["qa_readout_mask"] if run_qa else None,
                    answer_prefix_embeds=embedding(ids["qa_answer_prefix_ids"]) if run_qa else None,
                    answer_prefix_mask=ids["qa_answer_prefix_mask"] if run_qa else None,
                )
                task_ids = torch.cat(
                    [
                        torch.ones(
                            qa_input_ids.size(0), dtype=torch.long, device=device
                        ),
                        torch.zeros(
                            (ids["recon_target_ids"].size(0) if run_recon else 0),
                            dtype=torch.long,
                            device=device,
                        ),
                    ]
                )
                qa_probability, recon_probability = scheduled_task_weights(
                    step - stage_offset,
                    stage_total_steps,
                    cfg.data,
                    stage=stage if cfg.data.train_stage == "two_stage" else None,
                )
                if cfg.data.train_stage == "mix":
                    qa_rows, recon_rows = sample_task_masks(
                        ids["qa_context_indices"],
                        ids["recon_context_indices"],
                        qa_probability,
                        recon_probability,
                        cfg.training.seed + step,
                        ids["context_ids"].size(0),
                    )
                    row_mask = torch.cat([qa_rows, recon_rows])
                    qa_weight = recon_weight = 1.0
                else:
                    qa_weight, recon_weight = qa_probability, recon_probability
                    row_mask = None
                supervised, task_terms = supervision_objectives(
                    output.supervision_losses,
                    task_ids,
                    cfg.loss,
                    qa_weight=qa_weight * cfg.loss.qa_weight,
                    recon_weight=recon_weight * cfg.loss.recon_weight,
                    row_mask=(
                        output.supervision_mask
                        if row_mask is None
                        else output.supervision_mask & row_mask
                    ),
                )
                total = supervised
                terms = {
                    **task_terms,
                    "loss": total,
                    "qa_weight": total.detach() * 0 + qa_weight * cfg.loss.qa_weight,
                    "recon_weight": total.detach() * 0
                    + recon_weight * cfg.loss.recon_weight,
                    "qa_probability": total.detach() * 0 + qa_probability,
                    "recon_probability": total.detach() * 0 + recon_probability,
                }
                if accelerator is not None:
                    accelerator.backward(total)
                else:
                    # accelerate divides by gradient_accumulation_steps inside backward();
                    # the manual path has to do it here so the group is a true average.
                    (total / accumulation).backward()
                boundary = (
                    accelerator.sync_gradients if accelerator is not None else sync_now
                )
                for name, value in terms.items():
                    if isinstance(value, torch.Tensor):
                        detached = value.detach()
                        group_terms[name] = (
                            group_terms.get(name, detached.new_zeros(())) + detached
                        )
                group_micro_batches += 1
                if not boundary:
                    # Still accumulating: no optimizer step, no scheduler step, no evaluation.
                    continue
                log_step = (step + 1) % cfg.logging.log_every == 0
                if log_step:
                    named_parameters = list(base_model.named_parameters())
                    resampler_grad_norm = _gradient_norm(
                        parameter
                        for name, parameter in named_parameters
                        if name.startswith("context_resampler.")
                    )
                    metalora_grad_norm = _gradient_norm(
                        parameter
                        for name, parameter in named_parameters
                        if "lora_A" in name or "lora_B" in name
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
                active_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                # Every logged scalar is the mean over the micro-batches of this step, which
                # is what the optimizer step actually optimised.
                logged = {
                    name: float(value / group_micro_batches)
                    for name, value in group_terms.items()
                }
                group_terms.clear()
                micro_batches_in_step = group_micro_batches
                group_micro_batches = 0

                postfix = step_postfix(logged, micro_batches_in_step)
                progress.update(
                    progress_task,
                    advance=1,
                    status=" · ".join(
                        [
                            *(f"{name} {value}" for name, value in postfix.items()),
                            f"lr {active_scheduler.get_last_lr()[0]:.2e}",
                        ]
                    ),
                )

                if run and log_step:
                    # `train/lr` is logged next to the losses on purpose: the LR schedule is
                    # the one training-side scalar that silently went wrong before (accelerate
                    # stepping it once per rank, see the prepare() comment), and without this
                    # key a distorted schedule is invisible in the run history.
                    payload = {
                        "loss/total": logged["loss"],
                        "loss/qa": logged["qa_loss"],
                        "loss/recon": logged["recon_loss"],
                        "train/lr": float(active_scheduler.get_last_lr()[0]),
                        "loss/ppl": float(math.exp(min(logged["qa_loss"], 20.0))),
                        "loss/qa_ppl": float(math.exp(min(logged["qa_loss"], 20.0))),
                        "loss/recon_ppl": float(
                            math.exp(min(logged["recon_loss"], 20.0))
                        ),
                        "grad_norm/total": float(step_grad_norm.detach()),
                        "grad_norm/resampler": float(resampler_grad_norm),
                        "grad_norm/metalora": float(metalora_grad_norm),
                        "grad_norm/recon_task_embedding": float(
                            recon_task_embedding_grad_norm
                        ),
                        "data/qa_probability": float(qa_probability),
                        "data/recon_probability": float(recon_probability),
                        "data/micro_batches_per_step": float(micro_batches_in_step),
                    }
                    layer_weights = (
                        base_model.context_layer_logits.detach().float().softmax(dim=0)
                    )
                    payload.update(
                        {
                            f"train/alpha_layer_{layer_index}": float(weight)
                            for layer_index, weight in zip(
                                base_model.context_layer_indices,
                                layer_weights,
                                strict=True,
                            )
                        }
                    )
                    run.log(payload, step=step)
            is_main = accelerator is None or accelerator.is_main_process
            # Validation runs on EVERY rank.  accelerator.prepare shards the validation
            # loader, so evaluating only on the main process would (a) score one shard
            # instead of the whole split and (b) leave the other ranks waiting in the next
            # DDP collective for the entire evaluation, which the NCCL watchdog aborts
            # after 10 minutes ("Watchdog caught collective operation timeout").  The
            # evaluators all-reduce their accumulators, so every rank must call them.
            metrics = None
            # Final-step rule is centralized in should_evaluate(step, every, total).
            if len(validation_loader) and should_evaluate(
                step - stage_offset,
                cfg.evaluation.autoregressive_every,
                stage_total_steps,
            ):
                # Eval uses its own Rich live display; leave only one display active.
                progress.stop()
                checkpoint_model = (
                    accelerator.unwrap_model(model)
                    if accelerator is not None
                    else model
                )
                step_checkpoint = None
                if is_main:
                    CONSOLE.log(f"[cyan]Saving checkpoint[/cyan]  step-{step:,}.pt")
                    step_checkpoint = manager.save(
                        checkpoint_model,
                        optimizer,
                        scheduler,
                        step,
                        cfg.to_dict(),
                        swanlab_run_id=getattr(run, "id", None),
                        world_size=(
                            1 if accelerator is None else accelerator.num_processes
                        ),
                    )
                    CONSOLE.log(
                        f"[green]Checkpoint saved[/green]  {step_checkpoint.name} · last.pt refreshed"
                    )
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                if stage == "recon":
                    metrics = {
                        "recon_task": evaluator.recon_teacher_forced(
                            base_model,
                            validation_loader,
                            device,
                            row_loader=recon_row_loader,
                            context_base=recon_row_base,
                            pre_windowed=True,
                        )
                    }
                else:
                    metrics = evaluator.compare_reader_tasks(
                        base_model,
                        validation_loader,
                        device,
                        max_contexts_per_dataset=cfg.evaluation.max_eval_contexts_per_dataset,
                        max_qa_sample_per_context=cfg.data.max_qa_sample_per_context,
                        row_loader=validation_row_loader,
                        row_base=validation_row_base,
                        recon_row_loader=recon_row_loader,
                        recon_base=recon_row_base,
                        pre_windowed=True,
                        include_recon=cfg.evaluation.recon_teacherforce
                        or cfg.evaluation.recon_autoregressive,
                    )
                if is_main:
                    CONSOLE.print(
                        f"[bold green]Evaluation complete · step {step:,}[/bold green]"
                    )
                    CONSOLE.print(
                        format_evaluation_report(
                            metrics,
                            selection=validation_row_loader.evaluation_selection,
                            seed=cfg.training.seed,
                            step=step,
                            max_contexts=cfg.evaluation.max_eval_contexts_per_dataset,
                            max_new_tokens=cfg.evaluation.max_new_tokens,
                        )
                    )
            if run and is_main:
                for section, payload in eval_log_payloads(metrics):
                    for key in payload:
                        run.define_metric(key, section_name=section)
                    run.log(payload, step=step)
            if accelerator is not None:
                # Resynchronise explicitly so a rank whose shard finished early cannot
                # race ahead into the next training step.
                accelerator.wait_for_everyone()
            if metrics is not None and is_main:
                best_metric = (
                    metrics["recon_task"]["recon_token_accuracy"]
                    if stage == "recon"
                    else metrics["qa_task"]["f1"]
                )
                improved = manager.update_best(
                    step_checkpoint,
                    best_metric,
                )
                if improved:
                    CONSOLE.log(
                        f"[bold green]best.pt updated[/bold green]  step={step:,} · "
                        f"{stage} metric={best_metric:.4f}"
                    )
                else:
                    if manager.best_f1 is None:
                        CONSOLE.log(
                            f"[yellow]best.pt unchanged[/yellow]  "
                            f"{stage} metric={best_metric:.4f} is not finite"
                        )
                    else:
                        CONSOLE.log(
                            f"[dim]best.pt unchanged[/dim]  "
                            f"{stage} metric={best_metric:.4f} · "
                            f"best={manager.best_f1:.4f} at step {manager.best_step:,}"
                        )
            if metrics is not None:
                progress.start()
        if step >= total_steps:
            break
        stage_finished = cfg.data.train_stage == "two_stage" and (
            (stage == "recon" and epoch + 1 == cfg.training.epochs)
            or (stage == "qa" and epoch + 1 == total_epochs)
        )
        if stage_finished and (accelerator is None or accelerator.is_main_process):
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
                swanlab_run_id=getattr(run, "id", None),
                world_size=1 if accelerator is None else accelerator.num_processes,
            )
        if stage_finished and accelerator is not None:
            accelerator.wait_for_everyone()

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
            swanlab_run_id=getattr(run, "id", None),
            world_size=1 if accelerator is None else accelerator.num_processes,
        )
    progress.stop()
    if run:
        run.finish()
    return step, manager
