"""Record the hardware and software environment in results/environment.json."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import psutil

from inspection.artifacts import library_versions
from inspection.config import Settings


def _cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout.strip()
            if out:
                return out
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or "unknown"


def collect(settings: Settings) -> dict[str, Any]:
    import torch

    return {
        "cpu": _cpu_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "os": platform.platform(),
        "python": platform.python_version(),
        "libraries": library_versions(),
        "torch_threads": torch.get_num_threads(),
        "configured_threads": settings.threads,
        "seed": settings.seed,
        "gpu": "none (CPU only)",
    }


def write(settings: Settings) -> Path:
    out = settings.path("results_dir") / "environment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(collect(settings), indent=2))
    return out
