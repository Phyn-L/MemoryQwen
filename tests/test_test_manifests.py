"""CLI checks for selecting evaluation manifests by seed."""
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

SPEC = importlib.util.spec_from_file_location(
    "test_manifests_script", Path(__file__).resolve().parents[1] / "scripts/test_manifests.py"
)
test_manifests = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(test_manifests)


@pytest.mark.parametrize(
    ("seed_args", "expected"),
    [([], [42, 27, 72]), (["--seed", "27"], [27]), (["--seed", "72", "42"], [42, 72]),
     (["--seed", "42", "42"], [42]), (["--seed", "99"], None), (["--seed", "42", "99"], None)],
)
def test_cli_selects_manifest_seeds(tmp_path, monkeypatch, capsys, seed_args, expected):
    paths = []
    for index, seed in enumerate([42, 27, 72]):
        path = tmp_path / f"manifest_{index}.json"
        path.write_text(json.dumps({
            "seed": seed, "datasets": {}, "max_context_tokens": 128, "max_contexts": 1,
        }))
        paths.append(str(path))
    monkeypatch.setattr(test_manifests, "DEFAULT_MANIFESTS", paths)
    monkeypatch.setattr(test_manifests.shutil, "which", lambda _: "/mock/torchrun")
    output = tmp_path / "reports"
    argv = ["test_manifests", "--checkpoint", str(tmp_path / "model.pt"), "--output-dir", str(output)]
    # Exercise both default manifests and explicitly supplied paths.
    if seed_args:
        argv.extend(["--manifests", *paths])
    monkeypatch.setattr(sys, "argv", [*argv, *seed_args])
    launched = []

    def run(command, **kwargs):
        data_file = Path(command[command.index("--data-file") + 1])
        launched.append([42, 27, 72][int(data_file.stem.split("_")[1])])
        Path(command[command.index("--report-file") + 1]).write_text('{"qa_task": {}}')

    runner = Mock(side_effect=run)
    monkeypatch.setattr(test_manifests.subprocess, "run", runner)
    if expected is None:
        with pytest.raises(SystemExit) as error:
            test_manifests.main()
        assert error.value.code == 2
        assert "no manifests found for seeds: [99]" in capsys.readouterr().err
        runner.assert_not_called()
        assert not output.exists()
    else:
        test_manifests.main()
        assert launched == expected
        assert len(list(output.glob("*.report.json"))) == len(expected)
