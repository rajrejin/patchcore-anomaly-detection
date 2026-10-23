"""API settings: the `api` section of the shared config, overridable by MVI_API__* variables."""

from __future__ import annotations

from inspection.config import Settings, load_settings


def get_settings() -> Settings:
    return load_settings()
