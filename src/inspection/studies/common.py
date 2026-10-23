"""Shared machinery: fit a variant in memory, evaluate on the test split, resumable result files."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from inspection import metrics as M
from inspection.config import Settings
from inspection.coreset import greedy_kcenter, project_chunked, random_projection
from inspection.data import load_tensor, split_fit_calib, test_samples, train_paths
from inspection.features import PatchFeatureExtractor, require_real_weights
from inspection.latency import measure_latency
from inspection.patchcore import PatchCore, configure_runtime, extract_to_cache, score_paths
from inspection.threshold import calibrate

log = logging.getLogger(__name__)
PreFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class Variant:
    name: str
    backbone: str = "resnet50"
    pre_name: str = "none"
    pre: PreFn | None = None
    ratios: tuple[float, ...] = (0.1,)
    n_fit: int | None = None  # None means the whole fit split
    seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def results_file(settings: Settings, study: str, category: str) -> Path:
    d = settings.path("results_dir") / "studies" / study
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{category}.json"


def load_done(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def fit_models(settings: Settings, category: str, v: Variant) -> dict[float, PatchCore]:
    """Fit one memory bank per requested ratio. The greedy selection is nested, so one run at
    the largest ratio yields every smaller ratio as a prefix (identical to separate runs)."""
    require_real_weights(settings.backbone.weights)
    root = settings.path("data_dir")
    dc, pc = settings.data, settings.patchcore
    fit, calib = split_fit_calib(
        train_paths(root, category), v.seed, dc.calib_fraction, dc.min_calib
    )
    if v.n_fit is not None and v.n_fit < len(fit):
        rng = np.random.default_rng(v.seed)
        fit = [fit[i] for i in sorted(rng.permutation(len(fit))[: v.n_fit].tolist())]
    device = settings.resolve_device()
    ext = PatchFeatureExtractor(v.backbone, "imagenet", pc.pool_kernel, pc.reduce_dim).to(device)
    # the file name identifies the variant; the signature inside the cache also covers the model
    # settings (pooling, channel reduction, input size), so a config change re-extracts
    key = f"{v.backbone}_{v.pre_name}_s{v.seed}_n{len(fit)}"
    sig = key + json.dumps(settings.model_config_dict(), sort_keys=True)
    cache = settings.path("cache_dir") / category / "study" / f"{key}.npy"
    feats = extract_to_cache(
        ext,
        fit,
        cache,
        batch_size=pc.batch_size,
        resize=dc.resize,
        crop=dc.crop,
        key=sig,
        pre=v.pre,
        desc=f"{category}:{v.name}",
    )
    n = feats.shape[0]

    def k_for(r: float) -> int:
        return min(n, max(int(round(n * r)), min(n, pc.min_bank)))

    ks = {r: k_for(r) for r in v.ratios}
    proj = random_projection(feats.shape[1], pc.coreset_proj_dim, v.seed)
    z = project_chunked(feats, proj, device=device)
    order = greedy_kcenter(z, max(ks.values()), v.seed)
    if len(v.ratios) > 1:  # keep the selection order so CPU latency can be measured later
        np.save(
            results_file(settings, "ablations", category).with_name(
                f"{category}_{v.name}_order.npy"
            ),
            order,
        )
    models: dict[float, PatchCore] = {}
    for r, k in ks.items():
        idx = np.sort(order[:k])
        bank = torch.from_numpy(np.asarray(feats[idx], dtype=np.float32))
        m = PatchCore(
            ext,
            bank,
            knn_backend=pc.knn_backend,
            device=device,
            map_sigma=pc.map_sigma,
            crop=dc.crop,
        )
        cal, _ = score_paths(
            m, calib, batch_size=pc.batch_size, resize=dc.resize, crop=dc.crop, pre=v.pre
        )
        m.thresholds = calibrate(cal)
        models[r] = m
    return models


def evaluate_model(
    settings: Settings,
    category: str,
    model: PatchCore,
    pre: PreFn | None,
    *,
    pixel: bool = True,
    latency_n: int = 20,
) -> dict[str, Any]:
    from inspection.evaluate import load_cached_test

    if settings.resolve_device() != "cpu":
        latency_n = 0  # GPU latency says nothing about the CPU target; measured separately

    dc, pc = settings.data, settings.patchcore
    samples = test_samples(settings.path("data_dir"), category)
    cached = load_cached_test(settings, category)
    labels, masks = cached["test_labels"], cached["test_masks"]
    paths = [s.path for s in samples]
    scores, maps = score_paths(
        model, paths, batch_size=pc.batch_size, resize=dc.resize, crop=dc.crop, pre=pre,
        keep_maps=pixel,
    )  # fmt: skip
    thr = float(model.thresholds["image"])
    out: dict[str, Any] = {
        "image_auroc": M.image_auroc(labels, scores),
        "calibrated": M.threshold_metrics(labels, scores, thr),
        "threshold": thr,
        "bank_shape": list(model.bank.shape),
        "bank_mb": float(model.bank.numel() * 4 / 1024**2),
        "n_calib": int(model.thresholds["n_calib"]),
        "device": settings.resolve_device(),
    }
    if pixel and maps is not None:
        out["pixel_auroc"] = M.pixel_auroc(masks, maps)
    if latency_n:
        pick = np.linspace(0, len(paths) - 1, latency_n).astype(int)
        tens = [load_tensor(paths[i], dc.resize, dc.crop, pre) for i in pick]
        out["latency"] = measure_latency(model.score, tens, settings.latency.warmup)
    return out


def run_and_record(
    settings: Settings,
    study: str,
    category: str,
    variants: list[Variant],
    *,
    pixel: bool = True,
) -> dict[str, Any]:
    """Run variants not yet recorded; write after each so the study is resumable."""
    configure_runtime(settings)
    path = results_file(settings, study, category)
    done = load_done(path)
    for v in variants:
        names = {r: f"{v.name}@{r}" if len(v.ratios) > 1 else v.name for r in v.ratios}
        if all(n in done for n in names.values()):
            continue
        t = time.time()
        log.info("%s/%s: running %s", study, category, v.name)
        models = fit_models(settings, category, v)
        for r, m in models.items():
            res = evaluate_model(settings, category, m, v.pre, pixel=pixel)
            res.update(
                {
                    "variant": names[r],
                    "backbone": v.backbone,
                    "preprocessing": v.pre_name,
                    "coreset_ratio": r,
                    "n_fit": v.n_fit,
                    "seed": v.seed,
                    **v.extra,
                }
            )
            done[names[r]] = res
        done["_meta"] = {"seed_default": settings.seed}
        path.write_text(json.dumps(done, indent=2))
        log.info("%s/%s: %s done in %.0fs", study, category, v.name, time.time() - t)
    return done
