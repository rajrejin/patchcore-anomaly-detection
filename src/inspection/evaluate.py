"""Per-category evaluation on the MVTec AD test split."""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from inspection import metrics as M
from inspection.config import Settings
from inspection.data import MVTecDataset, Sample, load_tensor, test_samples
from inspection.latency import measure_latency, rss_mb
from inspection.patchcore import PatchCore, configure_runtime
from inspection.runtime_log import record_runtime

log = logging.getLogger(__name__)


def metric_bundle(
    labels: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray,
    maps: np.ndarray,
    thr_image: float,
    seed: int,
    with_pro: bool = True,
    with_ci: bool = True,
) -> dict[str, Any]:
    out: dict[str, Any] = {"image_auroc": M.image_auroc(labels, scores)}
    if with_ci:
        lo, hi = M.bootstrap_auroc_ci(labels, scores, 1000, seed)
        out["image_auroc_ci95"] = [lo, hi]
    out["pixel_auroc"] = M.pixel_auroc(masks, maps)
    if with_pro:
        out["au_pro"] = M.au_pro(masks, maps)
    out["calibrated"] = M.threshold_metrics(labels, scores, thr_image)
    out["oracle_best_f1"] = M.oracle_best_f1(labels, scores)
    return out


def run_test_set(
    model: PatchCore, samples: list[Sample], settings: Settings
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns labels (N,), scores (N,), masks (N,H,W) uint8, maps (N,H,W) float32."""
    ds = MVTecDataset(samples, settings.data.resize, settings.data.crop)
    bs = settings.patchcore.batch_size
    labels, scores, masks, maps = [], [], [], []
    for i in range(0, len(ds), bs):
        items = [ds[j] for j in range(i, min(i + bs, len(ds)))]
        x = torch.stack([it[0] for it in items])
        s, m = model.score(x)
        scores.append(s.numpy())
        maps.append(m.numpy())
        labels += [it[1] for it in items]
        masks += [it[2] for it in items]
    return np.array(labels), np.concatenate(scores), np.stack(masks), np.concatenate(maps)


def cache_dir(settings: Settings, category: str) -> Path:
    d = settings.path("cache_dir") / category
    d.mkdir(parents=True, exist_ok=True)
    return d


def _artifact_version(art_dir: Path) -> str | None:
    """Model version of the artifact on disk (manifest hash), without loading the bank."""
    from inspection.artifacts import manifest_version

    m = art_dir / "manifest.json"
    return manifest_version(json.loads(m.read_text())) if m.exists() else None


def _load_model(settings: Settings, category: str) -> PatchCore:
    from inspection.features import require_real_weights

    require_real_weights(settings.backbone.weights)
    configure_runtime(settings)
    return PatchCore.from_artifact(
        settings.path("artifacts_dir") / category,
        settings.patchcore.knn_backend,
        device=settings.resolve_device(),
    )


def score_test_set(
    settings: Settings, category: str, model: PatchCore | None = None
) -> tuple[list[Sample], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score the whole test split and store the arrays in cache/<category>/ and the per-image
    scores in results/scores/. Does not touch the headline results file."""
    model = model or _load_model(settings, category)
    samples = test_samples(settings.path("data_dir"), category)
    labels, scores, masks, maps = run_test_set(model, samples, settings)
    cd = cache_dir(settings, category)
    np.save(cd / "test_maps.npy", maps)
    np.save(cd / "test_masks.npy", masks)
    np.save(cd / "test_scores.npy", scores)
    np.save(cd / "test_labels.npy", labels)
    sdir = settings.path("results_dir") / "scores"
    sdir.mkdir(parents=True, exist_ok=True)
    with open(sdir / f"{category}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "defect_type", "label", "score"])
        for s, sc in zip(samples, scores, strict=True):
            w.writerow([f"{s.path.parent.name}/{s.path.name}", s.defect_type, s.label, f"{sc:.6f}"])
    return samples, labels, scores, masks, maps


def evaluate_category(settings: Settings, category: str, force: bool = False) -> dict[str, Any]:
    """Full evaluation: metrics, latency and memory into results/categories/<category>.json.

    Skips when a result exists for the same config and the same model version, so a retrained
    model (even with an unchanged config) is always re-evaluated."""
    out_file = settings.path("results_dir") / "categories" / f"{category}.json"
    art_dir = settings.path("artifacts_dir") / category
    if out_file.exists() and not force:
        prev = json.loads(out_file.read_text())
        if prev.get("config_hash") == settings.config_hash() and prev.get(
            "model_version"
        ) == _artifact_version(art_dir):
            log.info("%s: results up to date, skipping", category)
            return prev
    t_start = time.time()
    model = _load_model(settings, category)
    rss_loaded = rss_mb()
    samples, labels, scores, masks, maps = score_test_set(settings, category, model)
    thr = float(model.thresholds["image"])
    res = metric_bundle(labels, scores, masks, maps, thr, settings.seed)

    # latency: spread of normal and defective images, one at a time, model only
    n = min(settings.latency.max_images, len(samples))
    pick = np.linspace(0, len(samples) - 1, n).astype(int)
    tensors = [load_tensor(samples[i].path, settings.data.resize, settings.data.crop) for i in pick]
    lat = measure_latency(model.score, tensors, settings.latency.warmup)

    res.update(
        {
            "category": category,
            "seed": settings.seed,
            "config_hash": settings.config_hash(),
            "model_version": model.version,
            "n_test": len(samples),
            "n_test_good": int((labels == 0).sum()),
            "n_test_defective": int((labels == 1).sum()),
            "threshold_image": thr,
            "threshold_pixel": float(model.thresholds["pixel"]),
            "latency": lat,
            "rss_after_load_mb": rss_loaded,
            "bank_shape": list(model.bank.shape),
            "bank_mb": float(model.bank.numel() * 4 / 1024**2),
            "eval_seconds": time.time() - t_start,
        }
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(res, indent=2))
    record_runtime(settings, "evaluate_seconds", category, res["eval_seconds"])
    return res


def load_cached_test(settings: Settings, category: str) -> dict[str, np.ndarray]:
    """Test-split arrays for post-processing, the cascade and the studies. Re-scores the test
    split if the cache is missing, without rewriting the headline results."""
    cd = settings.path("cache_dir") / category
    need = ["test_maps", "test_masks", "test_scores", "test_labels"]
    if not all((cd / f"{n}.npy").exists() for n in need):
        score_test_set(settings, category)
    return {n: np.load(cd / f"{n}.npy") for n in need}
