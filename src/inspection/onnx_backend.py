"""ONNX export of the feature extractor, parity check and backend benchmarks."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from inspection.config import Settings
from inspection.data import load_tensor, test_samples
from inspection.features import PatchFeatureExtractor
from inspection.latency import PeakRSS, rss_mb

log = logging.getLogger(__name__)


def onnx_path(settings: Settings, backbone: str | None = None) -> Path:
    name = backbone or settings.backbone.name
    return settings.path("artifacts_dir") / "_onnx" / f"{name}_extractor.onnx"


class OnnxExtractor:
    """Drop-in for PatchFeatureExtractor.descriptors using ONNX Runtime on CPU."""

    def __init__(self, path: Path, threads: int = 4) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.weights_info = {"id": "onnx", "sha256": "unknown"}

    def descriptors(self, x: torch.Tensor) -> torch.Tensor:
        (out,) = self.session.run(None, {"image": x.numpy()})
        t = torch.from_numpy(out)
        return t.flatten(2).transpose(1, 2).contiguous()


def export_onnx(extractor: PatchFeatureExtractor, path: Path) -> Path:
    """Export backbone, pooling, upsampling and concatenation as one graph (dynamic batch)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, 224, 224)  # tracing input only; no results come from it
    torch.onnx.export(
        extractor,
        (dummy,),
        str(path),
        input_names=["image"],
        output_names=["features"],
        dynamic_axes={"image": {0: "batch"}, "features": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return path


def _time(fn: Any, args: list[Any], warmup: int = 3) -> dict[str, float]:
    for a in args[:warmup]:
        fn(a)
    ts, peak = [], 0.0
    with PeakRSS() as p:
        for a in args:
            t = time.perf_counter()
            fn(a)
            ts.append((time.perf_counter() - t) * 1000)
        peak = p.peak
    arr = np.asarray(ts)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
        "peak_rss_mb": peak,
    }


def export_and_benchmark(settings: Settings, category: str, n: int = 24) -> dict[str, Any]:
    from inspection.coreset import random_projection  # noqa: F401  (keeps imports honest)
    from inspection.knn import FaissKNN, TorchKNN
    from inspection.patchcore import PatchCore, configure_runtime

    configure_runtime(settings)
    model = PatchCore.from_artifact(settings.path("artifacts_dir") / category)
    ext = model.extractor
    path = export_onnx(ext, onnx_path(settings))
    ort_ext = OnnxExtractor(path, settings.threads)
    samples = test_samples(settings.path("data_dir"), category)
    pick = np.linspace(0, len(samples) - 1, n).astype(int)
    xs = [
        load_tensor(samples[i].path, settings.data.resize, settings.data.crop)[None] for i in pick
    ]

    diffs = []
    for x in xs[:8]:
        a = ext.descriptors(x)
        b = ort_ext.descriptors(x)
        diffs.append(float((a - b).abs().max()))
    out: dict[str, Any] = {
        "category": category,
        "seed": settings.seed,
        "onnx_file_mb": path.stat().st_size / 1024**2,
        "feature_parity_max_abs_diff": max(diffs),
        "parity_threshold": 1e-3,
        "parity_ok": max(diffs) < 1e-3,
        "rss_mb_before": rss_mb(),
    }
    out["feature_extractor"] = {
        "torch": _time(ext.descriptors, xs),
        "onnxruntime": _time(ort_ext.descriptors, xs),
    }
    descs = [ext.descriptors(x)[0] for x in xs]
    bank = model.bank
    knn_t = TorchKNN(bank)
    out["knn"] = {"torch_cdist": _time(knn_t.search, descs)}
    try:
        knn_f = FaissKNN(bank)
        out["knn"]["faiss"] = _time(knn_f.search, descs)
        diff = float((knn_t.search(descs[0]) - knn_f.search(descs[0])).abs().max())
        out["knn"]["max_abs_distance_diff"] = diff
    except ImportError:
        out["knn"]["faiss"] = "not installed"
    # end to end model-only latency for each combination
    combos: dict[str, Any] = {}
    for ename, e in (("torch", ext), ("onnx", ort_ext)):
        for kname, k in (("torch", knn_t), ("faiss", locals().get("knn_f"))):
            if k is None:
                continue

            def run(x: torch.Tensor, e: Any = e, k: Any = k) -> None:
                d = e.descriptors(x)
                k.search(d.reshape(-1, d.shape[-1]))

            combos[f"{ename}+{kname}"] = _time(run, xs)
    out["end_to_end_model_only"] = combos
    best = min(combos, key=lambda c: combos[c]["p50_ms"])
    out["fastest_by_p50"] = best
    dest = settings.path("results_dir") / "inference_benchmark.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    return out
