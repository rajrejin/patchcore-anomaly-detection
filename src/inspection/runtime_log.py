"""Measured run times in results/runtime.json.

Callers record a time only when they actually did the work, so re-running a resumable
command (which skips finished categories) never overwrites a real measurement.
"""

from __future__ import annotations

import json

from inspection.config import Settings


def record_runtime(settings: Settings, stage: str, category: str, seconds: float) -> None:
    f = settings.path("results_dir") / "runtime.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(f.read_text()) if f.exists() else {}
    data.setdefault(stage, {})[category] = round(seconds, 1)
    f.write_text(json.dumps(data, indent=2, sort_keys=True))
