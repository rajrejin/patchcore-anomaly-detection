"""Model service: upload validation, LRU of loaded categories, inference and rendering."""

from __future__ import annotations

import base64
import io
import json
import logging
import threading
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from inspection.artifacts import ArtifactError, manifest_version
from inspection.config import Settings
from inspection.data import CATEGORIES, geometry_image, normalize
from inspection.patchcore import PatchCore, configure_runtime
from inspection.postprocess import annotate, heatmap_overlay, process_map

log = logging.getLogger("api.service")
ALLOWED_FORMATS = {"PNG", "JPEG", "BMP"}


class ApiError(Exception):
    """An error with a stable code and HTTP status, safe to show to clients."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def decode_upload(data: bytes, settings: Settings) -> Image.Image:
    """Decode with Pillow (content, not content type), guarding size, pixels and dimensions."""
    a = settings.api
    if len(data) == 0:
        raise ApiError(400, "empty_upload", "the uploaded file is empty")
    if len(data) > a.max_upload_bytes:
        raise ApiError(413, "upload_too_large", f"upload exceeds {a.max_upload_bytes} bytes")
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = a.max_pixels  # decompression bomb guard, restored below
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            probe = Image.open(io.BytesIO(data))
            fmt = probe.format
            w, h = probe.size
            if fmt not in ALLOWED_FORMATS:
                raise ApiError(415, "unsupported_format", "use a PNG, JPEG or BMP image")
            if w * h > a.max_pixels:
                raise ApiError(413, "image_too_large", f"image exceeds {a.max_pixels} pixels")
            probe.verify()
            img = Image.open(io.BytesIO(data))
            img.load()
    except ApiError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ApiError(413, "image_too_large", "image exceeds the pixel limit") from exc
    except Exception as exc:  # noqa: BLE001 - any decode failure is a client error
        raise ApiError(400, "invalid_image", "the file is not a valid image") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
    w, h = img.size
    if min(w, h) < a.min_side or max(w, h) > a.max_side:
        raise ApiError(
            422, "bad_dimensions", f"image sides must be between {a.min_side} and {a.max_side} px"
        )
    return img.convert("RGB")


def png_b64(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@dataclass
class Stage1Data:
    fit: torch.Tensor
    threshold: float


class ModelService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.art_root = settings.path("artifacts_dir")
        self._lru: OrderedDict[str, PatchCore] = OrderedDict()
        self._stage1: dict[str, Stage1Data] = {}
        self._stage1_net: Any = None
        self._lock = threading.Lock()
        self.infos: dict[str, dict[str, Any]] = {}
        self.ready = False
        self.backends: dict[str, str] = {}
        configure_runtime(settings)

    # ------------------------------------------------------------ discovery
    def discover(self) -> None:
        """Read manifests only (cheap); full verification happens on load."""
        infos = {}
        for cat in CATEGORIES:
            m = self.art_root / cat / "manifest.json"
            if not m.exists():
                continue
            try:
                man = json.loads(m.read_text())
                infos[cat] = {
                    "category": cat,
                    "threshold": float(man["thresholds"]["image"]),
                    "model_version": manifest_version(man),
                    "created_at": man["created_at"],
                    "bank_shape": man["bank_shape"],
                    "cascade_available": (self.art_root / cat / "stage1.npz").exists(),
                }
            except (KeyError, ValueError, OSError):
                log.warning("skipping unreadable manifest for %s", cat)
        self.infos = infos

    def warmup(self) -> None:
        """Load up to LRU-size categories and run one forward pass so the first request is fast."""
        self.discover()
        for cat in list(self.infos)[: self.settings.api.lru_categories]:
            try:
                model = self.get_model(cat)
                model.score(torch.zeros(1, 3, 224, 224))
            except ArtifactError:
                log.exception("warm-up failed for %s", cat)
        self.ready = True

    # ------------------------------------------------------------ loading
    def get_model(self, category: str) -> PatchCore:
        if category not in CATEGORIES:
            raise ApiError(
                404, "unknown_category", "category is not one of the MVTec AD categories"
            )
        with self._lock:
            if category in self._lru:
                self._lru.move_to_end(category)
                return self._lru[category]
        art = self.art_root / category
        if not (art / "manifest.json").exists():
            raise ApiError(404, "category_not_trained", f"no trained artifact for {category}")
        try:
            model = self._load(art)
        except ArtifactError as exc:
            log.error("refusing artifact for %s: %s", category, exc)
            raise ApiError(
                503, "artifact_invalid", "the model artifact is missing or inconsistent"
            ) from exc
        with self._lock:
            self._lru[category] = model
            while len(self._lru) > self.settings.api.lru_categories:
                self._lru.popitem(last=False)
        return model

    def _load(self, art: Path) -> PatchCore:
        """Load with the configured serving backends, falling back to PyTorch when the ONNX
        file, onnxruntime or faiss is unavailable."""
        a = self.settings.api
        onnx_file: Path | None = None
        if a.extractor_backend == "onnx":
            from inspection.onnx_backend import onnx_path

            onnx_file = onnx_path(self.settings)
            if not onnx_file.exists():
                log.warning("ONNX file missing, using the PyTorch extractor")
                onnx_file = None
        for knn, onnx in ((a.knn_backend, onnx_file), ("torch", None)):
            try:
                model = PatchCore.from_artifact(
                    art, knn, onnx_file=onnx, threads=self.settings.threads
                )
                self.backends = {"knn": knn, "extractor": "onnx" if onnx else "torch"}
                return model
            except ImportError:
                log.warning("backend %s/%s unavailable, falling back", knn, onnx)
        raise ArtifactError("no usable backend")

    def get_stage1(self, category: str) -> tuple[Any, Stage1Data]:
        from inspection.cascade import Stage1

        f = self.art_root / category / "stage1.npz"
        if not f.exists():
            raise ApiError(409, "cascade_unavailable", f"no stage-1 data for {category}")
        if category not in self._stage1:
            z = np.load(f)
            q = self.settings.cascade.serve_q
            thr = float(np.percentile(z["calib_scores"], q))
            self._stage1[category] = Stage1Data(torch.from_numpy(z["fit"]), thr)
        if self._stage1_net is None:
            self._stage1_net = Stage1(self.settings)
        return self._stage1_net, self._stage1[category]

    # ------------------------------------------------------------ inference
    def inspect(self, category: str, img: Image.Image, use_cascade: bool) -> dict[str, Any]:
        s = self.settings
        t0 = time.perf_counter()
        model = self.get_model(category)
        rgb = geometry_image(img, s.data.resize, s.data.crop)
        x = normalize(rgb)[None]
        thr = float(model.thresholds["image"])
        stage1_score: float | None = None
        if use_cascade:
            net, st1 = self.get_stage1(category)
            tm = time.perf_counter()
            with torch.no_grad():
                stage1_score = float(torch.cdist(net.embed(x), st1.fit).min(1).values[0])
            if stage1_score <= st1.threshold:
                model_ms = (time.perf_counter() - tm) * 1000
                ann = annotate(rgb, [], stage1_score, anomalous=False)
                return {
                    "anomaly_score": None, "threshold": None, "is_anomalous": False,
                    "stage_reached": 1, "stage1_score": stage1_score, "boxes": [],
                    "heatmap": None, "annotated": ann, "model_ms": model_ms,
                    "end_to_end_ms": (time.perf_counter() - t0) * 1000,
                    "model_version": model.version,
                }  # fmt: skip
        tm = time.perf_counter()
        score_t, map_t = model.score(x)
        model_ms = (time.perf_counter() - tm) * 1000
        score, amap = float(score_t[0]), map_t[0].numpy()
        anomalous = score > thr
        boxes: list[tuple[int, int, int, int, float]] = []
        contours: list[np.ndarray] = []
        if anomalous:
            pp = s.postprocess
            r = process_map(amap, float(model.thresholds["pixel"]), pp.morph_kernel, pp.min_area)
            boxes, contours = r["boxes"], r["contours"]
        heat = heatmap_overlay(rgb, amap, thr)
        ann = annotate(rgb, boxes, score, contours, anomalous)
        return {
            "anomaly_score": score, "threshold": thr, "is_anomalous": anomalous,
            "stage_reached": 2, "stage1_score": stage1_score, "boxes": boxes,
            "heatmap": heat, "annotated": ann, "model_ms": model_ms,
            "end_to_end_ms": (time.perf_counter() - t0) * 1000,
            "model_version": model.version,
        }  # fmt: skip

    def loaded(self) -> list[str]:
        return list(self._lru)


def artifacts_available(root: Path) -> list[str]:
    return [c for c in CATEGORIES if (root / c / "manifest.json").exists()]
