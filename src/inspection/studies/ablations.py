"""Ablations: preprocessing, coreset ratio, backbone, and the seed spread of the baseline."""

from __future__ import annotations

import json
import logging

import numpy as np
import torch

from inspection.config import Settings
from inspection.data import load_tensor, test_samples
from inspection.latency import measure_latency
from inspection.patchcore import PatchCore, configure_runtime
from inspection.studies.common import Variant, results_file, run_and_record
from inspection.studies.preprocessing import PREPROCESSING

log = logging.getLogger(__name__)
SEEDS = (0, 1, 2)
RATIOS = (0.01, 0.1, 0.25)


def variants() -> list[Variant]:
    v: list[Variant] = []
    # coreset ratio at seed 0 (nested prefixes of one greedy run; the 0.1 entry equals the baseline)
    v.append(Variant("coreset", ratios=RATIOS, seed=0))
    for s in SEEDS[1:]:
        v.append(Variant(f"baseline_seed{s}", seed=s))
    for name, fn in PREPROCESSING.items():
        if name != "baseline":
            v.append(Variant(f"pre_{name}", pre_name=name, pre=fn, seed=0))
    v.append(Variant("backbone_wide_resnet50_2", backbone="wide_resnet50_2", seed=0))
    return v


def run_ablations(settings: Settings, categories: list[str]) -> None:
    for cat in categories:
        run_and_record(settings, "ablations", cat, variants())


def measure_ratio_latency(settings: Settings, categories: list[str]) -> None:
    """Measure CPU latency of the coreset-ratio banks on this machine.

    The studies may run on a GPU, where latency is meaningless for the CPU target. The study
    saves the greedy selection order; this rebuilds the 1, 10 and 25 percent banks from the
    local fit-split feature cache (same split, same seed) and times them on the CPU."""
    from inspection.features import PatchFeatureExtractor

    configure_runtime(settings)
    pc, dc = settings.patchcore, settings.data
    for cat in categories:
        order_file = results_file(settings, "ablations", cat).with_name(f"{cat}_coreset_order.npy")
        feats_file = settings.path("cache_dir") / cat / "fit_feats.npy"
        if not order_file.exists() or not feats_file.exists():
            log.warning("%s: missing %s or %s, skipping", cat, order_file.name, feats_file.name)
            continue
        order = np.load(order_file)
        feats = np.load(feats_file, mmap_mode="r")
        n = feats.shape[0]
        ext = PatchFeatureExtractor(
            settings.backbone.name, "imagenet", pc.pool_kernel, pc.reduce_dim
        )
        samples = test_samples(settings.path("data_dir"), cat)
        pick = np.linspace(0, len(samples) - 1, 20).astype(int)
        tens = [load_tensor(samples[i].path, dc.resize, dc.crop) for i in pick]
        path = results_file(settings, "ablations", cat)
        done = json.loads(path.read_text())
        for r in RATIOS:
            k = min(n, max(int(round(n * r)), min(n, pc.min_bank)))
            bank = torch.from_numpy(np.asarray(feats[np.sort(order[:k])], dtype=np.float32))
            model = PatchCore(ext, bank, knn_backend=pc.knn_backend, map_sigma=pc.map_sigma)
            lat = measure_latency(model.score, tens, settings.latency.warmup)
            entry = done[f"coreset@{r}"]
            entry["latency"] = lat
            entry["latency_note"] = "measured on the CPU after the study, banks rebuilt from order"
            log.info("%s ratio %s: p50 %.0f ms", cat, r, lat["p50_ms"])
        path.write_text(json.dumps(done, indent=2))
