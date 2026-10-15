"""Private-server routing and checkpoint identity across the tracking migration."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.checkpoint import CheckpointManager
from utils.config import TrainConfig
from utils.tracking import init_tracking


@pytest.fixture(autouse=True)
def clean_tracking_environment(monkeypatch):
    for name in ("SWANLAB_HOST", "SWANLAB_MODE", "SWANLAB_LOG_DIR", "SWANLAB_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_online_requires_explicit_private_host():
    with pytest.raises(ValueError, match="private SwanLab URL"):
        init_tracking(TrainConfig())


def test_private_host_and_resume_are_forwarded(monkeypatch, tmp_path):
    sdk = SimpleNamespace(login=Mock(), init=Mock(return_value=SimpleNamespace(id="existing-run")))
    monkeypatch.setitem(sys.modules, "swanlab", sdk)
    monkeypatch.setenv("SWANLAB_HOST", "http://private-server:8000")
    monkeypatch.setenv("SWANLAB_API_KEY", "private-key")
    cfg = TrainConfig()
    cfg.checkpoint.output_dir = str(tmp_path)
    cfg.logging.swanlab_host = "http://other-server:8000"
    cfg.logging.swanlab_run_id = "existing-run"
    init_tracking(cfg)
    sdk.login.assert_called_once_with(
        api_key="private-key", host="http://private-server:8000", relogin=True
    )
    args = sdk.init.call_args.kwargs
    assert args["id"] == "existing-run" and args["resume"] == "must"
    assert args["log_dir"] == str(tmp_path)
    assert args["config"]["logging"]["swanlab_log_dir"] == str(tmp_path)
    assert args["config"]["logging"]["swanlab_run_id"] == "existing-run"


def test_offline_never_logs_in_and_records_new_identity(monkeypatch):
    sdk = SimpleNamespace(login=Mock(), init=Mock(return_value=SimpleNamespace(id="new-run")))
    monkeypatch.setitem(sys.modules, "swanlab", sdk)
    monkeypatch.setenv("SWANLAB_MODE", "offline")
    cfg = TrainConfig()
    init_tracking(cfg)
    sdk.login.assert_not_called()
    assert sdk.init.call_args.kwargs["resume"] == "never"
    assert cfg.logging.swanlab_run_id == "new-run"


def test_disabled_skips_sdk(monkeypatch):
    monkeypatch.setenv("SWANLAB_MODE", "disabled")
    assert init_tracking(TrainConfig()) is None


def test_checkpoint_restores_swanlab_identity(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    manager = CheckpointManager(tmp_path)
    path = manager.save(model, optimizer, scheduler, 7, {}, swanlab_run_id="swanlab-id")
    assert manager.load(path, model) == (7, "swanlab-id")
