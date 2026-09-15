from pathlib import Path
import random
import torch

class CheckpointManager:
    def __init__(self, output_dir, save_every=1000):
        self.output_dir, self.save_every = Path(output_dir), save_every

    def save(self, model, optimizer, scheduler, step, config, final=False, wandb_run_id=None):
        path = self.output_dir / ("last.pt" if final else f"step-{step}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        trainable = {n: p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
        torch.save({"model": trainable, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step, "config": config, "wandb_run_id": wandb_run_id, "rng": {"python": random.getstate(), "torch": torch.get_rng_state()}}, path)
        return path

    def load(self, path, model, optimizer=None, scheduler=None):
        state = torch.load(path, map_location="cpu", weights_only=False)
        missing = model.load_state_dict(state.get("model", {}), strict=False)
        if optimizer is not None and state.get("optimizer") is not None: optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None: scheduler.load_state_dict(state["scheduler"])
        if state.get("rng"):
            random.setstate(state["rng"]["python"]); torch.set_rng_state(state["rng"]["torch"])
        return int(state.get("step", 0)), state.get("wandb_run_id")
