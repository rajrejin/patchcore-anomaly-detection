"""Two-stage cascade: cheap ResNet-18 embedding screen, then PatchCore on what remains."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from inspection import metrics as M
from inspection.config import Settings
from inspection.data import load_tensor, split_fit_calib, test_samples, train_paths
from inspection.features import global_embedding_extractor, require_real_weights
from inspection.latency import measure_latency
from inspection.patchcore import configure_runtime

log = logging.getLogger(__name__)


class Stage1:
    """ResNet-18 at 128 x 128, global-average-pooled embedding, nearest fit embedding distance."""

    def __init__(self, settings: Settings, fit_embeddings: torch.Tensor | None = None) -> None:
        self.size = settings.cascade.input_size
        self.net = global_embedding_extractor(settings.cascade.backbone, settings.backbone.weights)
        self.fit = fit_embeddings

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", antialias=True)
        return self.net(x)

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> torch.Tensor:
        assert self.fit is not None
        return torch.cdist(self.embed(x), self.fit).min(1).values

    def embed_paths(self, paths: list[Path], settings: Settings, batch: int = 32) -> torch.Tensor:
        out = []
        for i in range(0, len(paths), batch):
            x = torch.stack(
                [
                    load_tensor(p, settings.data.resize, settings.data.crop)
                    for p in paths[i : i + batch]
                ]
            )
            out.append(self.embed(x))
        return torch.cat(out)


def run_cascade(settings: Settings, category: str) -> dict[str, Any]:
    from inspection.evaluate import load_cached_test

    require_real_weights(settings.backbone.weights)
    configure_runtime(settings)
    root = settings.path("data_dir")
    base = json.loads(
        (settings.path("results_dir") / "categories" / f"{category}.json").read_text()
    )
    d = load_cached_test(settings, category)
    labels, s2 = d["test_labels"].astype(int), d["test_scores"]
    thr2 = base["threshold_image"]

    fit, calib = split_fit_calib(
        train_paths(root, category),
        settings.seed,
        settings.data.calib_fraction,
        settings.data.min_calib,
    )
    st1 = Stage1(settings)
    st1.fit = st1.embed_paths(fit, settings)
    cal_scores = torch.cdist(st1.embed_paths(calib, settings), st1.fit).min(1).values.numpy()
    art_dir = settings.path("artifacts_dir") / category
    if art_dir.exists():
        np.savez_compressed(
            art_dir / "stage1.npz",
            fit=st1.fit.numpy(),
            calib_scores=cal_scores,
            backbone=np.array(settings.cascade.backbone),
            input_size=np.array(settings.cascade.input_size),
        )
    samples = test_samples(root, category)
    s1 = (
        torch.cdist(st1.embed_paths([s.path for s in samples], settings), st1.fit)
        .min(1)
        .values.numpy()
    )

    # measured per-stage latency (model only, one image at a time)
    n = min(settings.latency.max_images, len(samples))
    pick = np.linspace(0, len(samples) - 1, n).astype(int)
    tens = [load_tensor(samples[i].path, settings.data.resize, settings.data.crop) for i in pick]
    lat1 = measure_latency(st1.score, tens, settings.latency.warmup)
    t1, t2 = lat1["mean_ms"], base["latency"]["mean_ms"]

    base_pred = s2 > thr2
    base_m = M.threshold_metrics(labels, s2, thr2)
    rows = []
    for q in settings.cascade.quantiles:
        thr1 = -np.inf if q == 0 else float(np.percentile(cal_scores, q))
        skipped = s1 <= thr1
        final_scores = np.where(skipped, s2.min() - 1.0, s2)
        pred = (~skipped) & base_pred
        m = M.threshold_metrics(labels, np.where(pred, 1.0, 0.0), 0.5)
        skip_n = float(skipped[labels == 0].mean())
        skip_d = float(skipped[labels == 1].mean())
        mean_lat = t1 + (1 - float(skipped.mean())) * t2 if q > 0 else t2
        mix = {}
        for rate in settings.cascade.defect_rates:
            skip_rate = (1 - rate) * skip_n + rate * skip_d
            lat = t1 + (1 - skip_rate) * t2 if q > 0 else t2
            mix[str(rate)] = {
                "skip_rate": skip_rate,
                "mean_latency_ms": lat,
                "speedup": t2 / lat,
                "defects_wrongly_skipped_per_1000_items": 1000 * rate * skip_d,
            }
        rows.append(
            {
                "q": q,
                "stage1_threshold": None if q == 0 else thr1,
                "normal_skipped": skip_n,
                "defective_wrongly_skipped": skip_d,
                "image_auroc": M.image_auroc(labels, final_scores),
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "mean_latency_ms": mean_lat,
                "speedup": t2 / mean_lat,
                "production_mix_calculation": mix,
            }
        )
    out = {
        "category": category,
        "seed": settings.seed,
        "stage1_auroc_alone": M.image_auroc(labels, s1),
        "stage1_latency_ms_mean": t1,
        "stage2_latency_ms_mean": t2,
        "patchcore_alone": {**base_m, "image_auroc": M.image_auroc(labels, s2)},
        "latency_note": "mean latency is computed from measured per-stage means and the "
        "measured skip fractions, not from timing the combined pipeline",
        "rows": rows,
    }
    odir = settings.path("results_dir") / "cascade"
    odir.mkdir(parents=True, exist_ok=True)
    (odir / f"{category}.json").write_text(json.dumps(out, indent=2))
    return out
