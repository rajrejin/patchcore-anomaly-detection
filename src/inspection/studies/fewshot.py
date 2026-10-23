"""Few-shot study: memory bank built from N normal images."""

from __future__ import annotations

from inspection.config import Settings
from inspection.studies.common import Variant, run_and_record

# The "all" point is taken from the main run (seed 0) and the ablation baselines (seeds 1, 2).
NS = (1, 5, 10, 25, 50)
SEEDS = (0, 1, 2)


def run_fewshot(settings: Settings, categories: list[str]) -> None:
    for cat in categories:
        vs = [
            Variant(f"n{n}_seed{s}", n_fit=n, seed=s, extra={"n_label": n})
            for n in NS
            for s in SEEDS
        ]
        run_and_record(settings, "fewshot", cat, vs, pixel=False)
