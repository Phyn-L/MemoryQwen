import math
import os
import random
import shutil
from pathlib import Path

import torch


def _capture_rng_state():
    """Python, NumPy, torch CPU and every CUDA device RNG state.

    The torch *CPU* state alone is not enough. ``losses.sample_positions`` draws on the device
    RNG (CUDA during training) and dropout uses it too, so a resumed run would otherwise
    sample a different distillation subset -- and different dropout masks -- than the
    uninterrupted run it is continuing, with ``deterministic: true`` unable to help because
    deterministic algorithms only fix the *operators*, not the random stream.
    """
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is optional for this project
        pass
    else:
        state["numpy"] = np.random.get_state()
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if state.get("numpy") is not None:
        try:
            import numpy as np
        except ImportError:  # pragma: no cover
            pass
        else:
            np.random.set_state(state["numpy"])
    cuda = state.get("torch_cuda")
    if cuda and torch.cuda.is_available():
        # A checkpoint written on an 8-GPU node cannot restore 8 device generators on a
        # 4-GPU one; resuming with a different device count is already a different run (the
        # sampler and the data sharding differ too), so keep the devices that do match.
        if len(cuda) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda)


class CheckpointManager:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.best_f1 = None
        self.best_step = None
        # Metadata of the most recent ``load``. A resumed run needs the step (where to pick
        # up in the data stream) and the rank count it was written with: the sampler is
        # deterministic *given the same sharding*, so a different world size would shift
        # every rank's batches and a mid-epoch resume would silently replay other data.
        self.loaded = {}

    def save(self, model, optimizer, scheduler, step, config, final=False,
             aim_run_id=None, world_size=1, validation_f1=None):
        path = self.output_dir / ("last.pt" if final else f"step-{step}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Only trainable tensors are stored: the frozen Qwen backbone is ~1.7B parameters
        # that never change, so saving it would multiply checkpoint size by roughly ten.
        trainable = {n: p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
        state = {
            "model": trainable,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "config": config,
            "aim_run_id": aim_run_id,
            "world_size": int(world_size),
            "rng": _capture_rng_state(),
        }
        score = None if validation_f1 is None else float(validation_f1)
        improved = score is not None and math.isfinite(score) and (
            self.best_f1 is None or score > self.best_f1
        )
        best_f1 = score if improved else self.best_f1
        best_step = step if improved else self.best_step
        state.update(validation_f1=score, best_f1=best_f1, best_step=best_step)
        # Replacing the inode preserves any historical checkpoint linked to last/best.
        temporary = path.with_suffix(".pt.tmp")
        try:
            torch.save(state, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        if improved:
            self._refresh_alias(path, "best.pt")
        self.best_f1, self.best_step = best_f1, best_step
        if not final:
            self._refresh_alias(path, "last.pt")
        checkpoints = sorted(
            (p for p in self.output_dir.glob("step-*.pt") if p.stem[5:].isdigit()),
            key=lambda p: int(p.stem[5:]),
        )
        for old in checkpoints[:-3]:
            old.unlink()
        return path

    def _refresh_alias(self, source, name):
        target = self.output_dir / name
        temporary = target.with_suffix(".pt.tmp")
        temporary.unlink(missing_ok=True)
        try:
            try:
                os.link(source, temporary)
            except OSError:
                shutil.copy2(source, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self, path, model, optimizer=None, scheduler=None, allow_missing_trainable=False, restore_rng=True):
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
                "A shape-defining config value (memory_length, lora_rank, target_modules) differs "
                "from the run that produced it."
            ) from exc

        if unexpected:
            raise RuntimeError(
                f"checkpoint {path} holds {len(unexpected)} tensors this model has no place for, "
                f"e.g. {sorted(unexpected)[:8]}. The architecture or the objective changed "
                "(for example the objective or model architecture changed)."
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

        if optimizer is not None and state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        if restore_rng and state.get("rng"):
            _restore_rng_state(state["rng"])
        self.best_f1 = state.get("best_f1")
        self.best_step = state.get("best_step")
        self.loaded = {
            "step": int(state.get("step", 0)),
            "world_size": int(state.get("world_size", 1) or 1),
            "config": state.get("config") or {},
        }
        return self.loaded["step"], state.get("aim_run_id")
