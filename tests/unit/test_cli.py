"""The mvinspect command line: argument handling and the commands that need no model."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from inspection.cli import app

runner = CliRunner()


def plain(text: str) -> str:
    """Drop terminal colour codes; on CI the help text is coloured, which splits option names."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_help_lists_every_command() -> None:
    out = plain(runner.invoke(app, ["--help"]).output)
    for cmd in ("fetch-data", "verify-data", "explore", "train", "evaluate", "cascade", "ablate",
                "studies", "export-onnx", "serve", "report"):  # fmt: skip
        assert cmd in out


def test_unknown_category_is_rejected() -> None:
    res = runner.invoke(app, ["train", "not_a_category"])
    assert res.exit_code != 0
    assert "unknown categories" in plain(res.output)


def test_train_needs_categories_or_all() -> None:
    res = runner.invoke(app, ["train"])
    assert res.exit_code != 0
    assert "--all" in plain(res.output)


def test_verify_data_fails_on_empty_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MVI_PATHS__DATA_DIR", str(tmp_path))
    res = runner.invoke(app, ["verify-data"])
    assert res.exit_code == 1
    assert "missing category directory" in res.output


def test_env_writes_environment_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MVI_PATHS__RESULTS_DIR", str(tmp_path))
    res = runner.invoke(app, ["env"])
    assert res.exit_code == 0, res.output
    env = json.loads((tmp_path / "environment.json").read_text())
    assert env["seed"] == 0 and env["physical_cores"] >= 1
