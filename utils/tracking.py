"""Main-process experiment tracking against a private SwanLab instance."""

import os

from utils.config import TrainConfig


def init_tracking(cfg: TrainConfig):
    logging = cfg.logging
    logging.swanlab_log_dir = cfg.checkpoint.output_dir
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
        log_dir=logging.swanlab_log_dir,
        id=logging.swanlab_run_id,
        resume="must" if logging.swanlab_run_id else "never",
    )
    logging.swanlab_run_id = run.id
    run.define_metric("eval_qa/*", section_name="eval_qa")
    run.define_metric("eval_recon/*", section_name="eval_recon")
    return run
