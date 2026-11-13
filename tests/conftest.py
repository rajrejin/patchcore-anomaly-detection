from __future__ import annotations

from pathlib import Path

import pytest

from inspection.config import load_settings


@pytest.fixture(scope="session")
def settings():  # type: ignore[no-untyped-def]
    return load_settings()


@pytest.fixture(scope="session")
def data_root(settings) -> Path:  # type: ignore[no-untyped-def]
    root = settings.path("data_dir")
    if not (root / "bottle" / "test").is_dir():
        pytest.skip("MVTec AD not found in data/mvtec_ad (run `mvinspect fetch-data`)")
    return root


@pytest.fixture(scope="session")
def bottle_artifact(settings) -> Path:  # type: ignore[no-untyped-def]
    art = settings.path("artifacts_dir") / "bottle"
    if not (art / "manifest.json").exists():
        pytest.skip("trained bottle artifact not found (run: mvinspect train bottle)")
    return art
