from pathlib import Path
import os
import random
import shutil
import torch

class CheckpointManager:
    def __init__(self, output_dir, save_every=1000):
        self.output_dir, self.save_every = Path(output_dir), save_every
        # Metadata of the most recent ``load``. A resumed run needs the step (where to pick
        # up in the data stream) and the rank count it was written with: the sampler is
        # deterministic *given the same sharding*, so a different world size would shift
        # every rank's batches and a mid-epoch resume would silently replay other data.
        self.loaded = {}

    def save(self, model, optimizer, scheduler, step, config, final=False,
             wandb_run_id=None, world_size=1):
        path = self.output_dir / ("last.pt" if final else f"step-{step}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Only trainable tensors are stored: the frozen Qwen backbone is ~1.7B parameters
        # that never change, so saving it would multiply checkpoint size by roughly ten.
        trainable = {n: p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
        state = {"model": trainable, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step, "config": config, "wandb_run_id": wandb_run_id, "world_size": int(world_size), "rng": {"python": random.getstate(), "torch": torch.get_rng_state()}}
        torch.save(state, path)
        if not final:
            # Refresh ``last.pt`` at every periodic save too. It used to be written only by
            # the final save, so a crash between two intervals left no last.pt at all and
            # "resume from the newest checkpoint" had to glob step-*.pt. Hard-link when the
            # filesystem allows it (no second copy of a ~1 GiB file), else copy.
            last = self.output_dir / "last.pt"
            try:
                last.unlink()
            except FileNotFoundError:
                pass
            try:
                os.link(path, last)
            except OSError:
                shutil.copy2(path, last)
        return path

    def load(self, path, model, optimizer=None, scheduler=None, allow_missing_trainable=False):
        """Restore the trainable subset of ``model`` from ``path``.

        ``strict=False`` is required because the frozen backbone is deliberately absent
        from the file. On its own it is silent though: a checkpoint that predates a new
        module, or that was written with a different ``memory_length`` / ``lora_rank`` /
        decoder width, would load "successfully" and leave part of the model at its random
        initialisation. The checks below reject exactly those cases and still allow the
        frozen backbone to be missing, which is the intended save format.
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        saved = state.get("model", {})
        try:
            missing, unexpected = model.load_state_dict(saved, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                f"checkpoint {path} does not fit this model: {exc}\n"
                "A shape-defining config value (memory_length, lora_rank, "
                "decoder_hidden_size/decoder_heads/decoder_ffn_ratio, target_modules) differs "
                "from the run that produced it."
            ) from exc

        if unexpected:
            raise RuntimeError(
                f"checkpoint {path} holds {len(unexpected)} tensors this model has no place for, "
                f"e.g. {sorted(unexpected)[:8]}. The architecture or the objective changed "
                "(for example reconstruction_loss switched between context_lm and mse_cosine)."
            )

        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        absent = sorted(trainable.intersection(missing))
        if absent:
            detail = (
                f"checkpoint {path} is missing {len(absent)} of {len(trainable)} trainable tensors, "
                f"e.g. {absent[:8]}. They would silently keep their random initialisation."
            )
            if not allow_missing_trainable:
                raise RuntimeError(
                    detail + " Pass allow_missing_trainable=True to load the rest anyway "
                    "(useful when only initialising from an older run)."
                )
            print(f"[checkpoint] WARNING: {detail}")

        if optimizer is not None and state.get("optimizer") is not None: optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None: scheduler.load_state_dict(state["scheduler"])
        if state.get("rng"):
            random.setstate(state["rng"]["python"]); torch.set_rng_state(state["rng"]["torch"])
        self.loaded = {
            "step": int(state.get("step", 0)),
            "world_size": int(state.get("world_size", 1) or 1),
            "config": state.get("config") or {},
        }
        return self.loaded["step"], state.get("wandb_run_id")
