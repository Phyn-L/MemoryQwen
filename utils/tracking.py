"""Main-process experiment tracking against a private SwanLab instance."""

import os
from pathlib import Path

from utils.config import TrainConfig


def init_tracking(cfg: TrainConfig):
    logging = cfg.logging
    mode = os.environ.get("SWANLAB_MODE") or logging.swanlab_mode
    if mode not in {"online", "offline", "local", "disabled"}:
        raise ValueError(f"Invalid SwanLab mode: {mode!r}")
    if mode == "disabled":
        return None
    host = os.environ.get("SWANLAB_HOST") or logging.swanlab_host
    if mode == "online" and not host:
        raise ValueError(
            "Set SWANLAB_HOST or logging.swanlab_host to your private SwanLab URL; "
            "use logging.swanlab_mode: offline to record without a server."
        )
    import swanlab

    if mode == "online":
        # Explicit host and re-authentication prevent reusing an official-cloud session.
        swanlab.login(api_key=os.environ.get("SWANLAB_API_KEY"), host=host, relogin=True)
    run = swanlab.init(
        project=logging.swanlab_project,
        workspace=logging.swanlab_workspace,
        name=logging.swanlab_run_name,
        config=cfg.to_dict(),
        mode=mode,
        log_dir=os.environ.get("SWANLAB_LOG_DIR")
        or logging.swanlab_log_dir
        or str(Path(cfg.checkpoint.output_dir) / "swanlog"),
        id=logging.swanlab_run_id,
        resume="must" if logging.swanlab_run_id else "never",
    )
    logging.swanlab_run_id = run.id
    return run
