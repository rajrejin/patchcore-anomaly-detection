"""Robustness: perturb test images; model and calibrated threshold stay fixed."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable

import cv2
import numpy as np

from inspection import metrics as M
from inspection.config import Settings
from inspection.data import test_samples
from inspection.patchcore import PatchCore, configure_runtime, score_paths
from inspection.studies.common import load_done, results_file

log = logging.getLogger(__name__)
PreFn = Callable[[np.ndarray], np.ndarray]


def brightness(factor: float) -> PreFn:
    return lambda im: np.clip(im.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def noise(sigma: float, seed: int = 0) -> PreFn:
    rng = np.random.default_rng(seed)
    return lambda im: np.clip(im + rng.normal(0, sigma, im.shape), 0, 255).astype(np.uint8)


def blur(sigma: float) -> PreFn:
    return lambda im: cv2.GaussianBlur(im, (0, 0), sigma)


def jpeg(quality: int) -> PreFn:
    def f(im: np.ndarray) -> np.ndarray:
        ok, buf = cv2.imencode(
            ".jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        assert ok
        dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        assert dec is not None
        return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

    return f


def affine(angle: float = 0.0, shift: float = 0.0) -> PreFn:
    def f(im: np.ndarray) -> np.ndarray:
        h, w = im.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        m[0, 2] += shift
        return cv2.warpAffine(im, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return f


def perturbations() -> list[tuple[str, str, PreFn | None]]:
    p: list[tuple[str, str, PreFn | None]] = [("none", "-", None)]
    p += [("brightness", f"x{f}", brightness(f)) for f in (0.6, 0.8, 1.2, 1.4)]
    p += [("gaussian_noise", f"sigma{s}", noise(s)) for s in (5, 10, 20)]
    p += [("gaussian_blur", f"sigma{s}", blur(s)) for s in (1, 2, 3)]
    p += [("jpeg", f"q{q}", jpeg(q)) for q in (60, 30)]
    p += [("rotation", f"{a}deg", affine(angle=a)) for a in (2, 5, 10)]
    p += [("translation", f"{s}px", affine(shift=s)) for s in (4, 8, 16)]
    return p


def run_robustness(settings: Settings, categories: list[str]) -> None:
    configure_runtime(settings)
    dc, pc = settings.data, settings.patchcore
    for cat in categories:
        path = results_file(settings, "robustness", cat)
        done = load_done(path)
        model = PatchCore.from_artifact(
            settings.path("artifacts_dir") / cat, pc.knn_backend, device=settings.resolve_device()
        )
        thr = float(model.thresholds["image"])
        samples = test_samples(settings.path("data_dir"), cat)
        labels = np.array([s.label for s in samples])
        paths = [s.path for s in samples]
        for kind, severity, fn in perturbations():
            key = f"{kind}:{severity}"
            if key in done:
                continue
            scores, _ = score_paths(
                model, paths, batch_size=pc.batch_size, resize=dc.resize, crop=dc.crop, pre=fn
            )
            m = M.threshold_metrics(labels, scores, thr)
            done[key] = {
                "perturbation": kind,
                "severity": severity,
                "image_auroc": M.image_auroc(labels, scores),
                "fpr_normals": m["fpr"],
                "recall": m["recall"],
                "f1": m["f1"],
                "threshold": thr,
                "seed": settings.seed,
            }
            path.write_text(json.dumps(done, indent=2))
            log.info("robustness %s %s auroc %.3f", cat, key, done[key]["image_auroc"])
