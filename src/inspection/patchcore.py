"""PatchCore: memory bank of coreset patch descriptors, nearest-neighbour anomaly scoring."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from inspection.artifacts import ArtifactError, load_artifact, manifest_version, save_artifact
from inspection.config import Settings
from inspection.coreset import coreset_indices
from inspection.data import dataset_fingerprint, load_tensor, split_fit_calib, train_paths
from inspection.features import PatchFeatureExtractor, require_real_weights
from inspection.knn import make_knn
from inspection.logging_utils import seed_everything
from inspection.runtime_log import record_runtime
from inspection.threshold import calibrate

log = logging.getLogger(__name__)

PreFn = Callable[[np.ndarray], np.ndarray]


def configure_runtime(settings: Settings) -> None:
    torch.set_num_threads(settings.threads)
    seed_everything(settings.seed)


def build_extractor(settings: Settings) -> PatchFeatureExtractor:
    return PatchFeatureExtractor(
        settings.backbone.name,
        settings.backbone.weights,
        settings.patchcore.pool_kernel,
        settings.patchcore.reduce_dim,
    ).to(settings.resolve_device())


def gaussian_kernel1d(sigma: float, radius: int | None = None) -> torch.Tensor:
    radius = radius if radius is not None else int(4 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def gaussian_blur(maps: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of (B, H, W) maps with replicate padding."""
    if sigma <= 0:
        return maps
    k = gaussian_kernel1d(sigma)
    r = k.numel() // 2
    x = maps[:, None]
    x = F.pad(x, (r, r, 0, 0), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, r, r), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, -1, 1))
    return x[:, 0]


def to_anomaly_map(patch_scores: torch.Tensor, out_size: int, sigma: float) -> torch.Tensor:
    """(B, h, w) patch scores -> (B, out, out): bilinear upsample then Gaussian blur."""
    up = F.interpolate(
        patch_scores[:, None], size=(out_size, out_size), mode="bilinear", align_corners=False
    )
    return gaussian_blur(up[:, 0], sigma)


class PatchCore:
    def __init__(
        self,
        extractor: Any,
        bank: torch.Tensor,
        *,
        knn_backend: str = "torch",
        device: str = "cpu",
        map_sigma: float = 4.0,
        crop: int = 224,
        thresholds: dict[str, Any] | None = None,
        manifest: dict[str, Any] | None = None,
    ) -> None:
        self.extractor = extractor
        self.bank = bank
        self.device = device
        self.knn = make_knn(knn_backend, bank, device)
        self.map_sigma, self.crop = map_sigma, crop
        self.thresholds = thresholds or {}
        self.manifest = manifest or {}

    @torch.no_grad()
    def patch_scores(self, x: torch.Tensor) -> torch.Tensor:
        """(B, h, w) nearest-neighbour distance for every patch."""
        descriptors = self.extractor.descriptors(x)
        batch, n_patches, dim = descriptors.shape
        # One long list of patches goes into the search; distances come back in the same order.
        distances = self.knn.search(descriptors.reshape(batch * n_patches, dim)).cpu()
        side = int(round(n_patches**0.5))  # patches form a square grid, e.g. 784 -> 28 x 28
        return distances.reshape(batch, side, side)

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Image scores (max patch score) and anomaly maps (B, crop, crop)."""
        patch_scores = self.patch_scores(x)
        image_scores = patch_scores.flatten(1).max(1).values  # the most unusual patch decides
        return image_scores, to_anomaly_map(patch_scores, self.crop, self.map_sigma)

    @property
    def version(self) -> str:
        return manifest_version(self.manifest) if self.manifest else "unversioned"

    @classmethod
    def from_artifact(
        cls,
        art_dir: Path,
        knn_backend: str = "torch",
        verify_hash: bool = True,
        onnx_file: Path | None = None,
        threads: int = 4,
        device: str = "cpu",
    ) -> PatchCore:
        """Load and verify an artifact. Pass `onnx_file` to run the feature extractor with ONNX
        Runtime instead of PyTorch (the PyTorch extractor is still built to check weights)."""
        bank, manifest = load_artifact(art_dir, verify_hash)
        mc = manifest["model_config"]
        ext = PatchFeatureExtractor(
            mc["backbone"]["name"],
            mc["backbone"]["weights"],
            mc["patchcore"]["pool_kernel"],
            mc["patchcore"]["reduce_dim"],
        )
        want = manifest["backbone"].get("sha256")
        have = ext.weights_info["sha256"]
        if (
            want not in (None, "unknown", "none")
            and have not in ("unknown", "none")
            and want != have
        ):
            raise ArtifactError("backbone weights checksum differs from the artifact manifest")
        runner: Any = ext.to(device)
        if onnx_file is not None:
            from inspection.onnx_backend import OnnxExtractor

            runner = OnnxExtractor(onnx_file, threads)
        return cls(
            runner,
            bank,
            knn_backend=knn_backend,
            device=device,
            map_sigma=mc["patchcore"]["map_sigma"],
            crop=mc["data"]["crop"],
            thresholds=manifest["thresholds"],
            manifest=manifest,
        )


# ------------------------------------------------------------------ training


def _paths_hash(paths: list[Path], extra: str) -> str:
    h = hashlib.sha256(extra.encode())
    for p in paths:
        h.update(f"{p.name}:{p.stat().st_size}\n".encode())
    return h.hexdigest()[:16]


def extract_to_cache(
    extractor: PatchFeatureExtractor,
    paths: list[Path],
    cache_file: Path,
    *,
    batch_size: int,
    resize: int,
    crop: int,
    key: str,
    pre: PreFn | None = None,
    desc: str = "features",
) -> np.ndarray:
    """Write descriptors as a float16 .npy memmap of shape (N*P, D). Reuses a matching cache."""
    meta_file = cache_file.with_suffix(".json")
    sig = _paths_hash(paths, key)
    if cache_file.exists() and meta_file.exists():
        meta = json.loads(meta_file.read_text())
        if meta.get("sig") == sig and meta.get("complete"):
            return np.load(cache_file, mmap_mode="r")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    out: Any = None
    row = 0
    for i in tqdm(range(0, len(paths), batch_size), desc=desc, leave=False):
        chunk = paths[i : i + batch_size]
        x = torch.stack([load_tensor(p, resize, crop, pre) for p in chunk])
        d = extractor.descriptors(x)
        n, pp, dim = d.shape
        if out is None:
            out = np.lib.format.open_memmap(
                cache_file, mode="w+", dtype=np.float16, shape=(len(paths) * pp, dim)
            )
        out[row : row + n * pp] = d.reshape(n * pp, dim).cpu().numpy().astype(np.float16)
        row += n * pp
    assert out is not None
    out.flush()
    meta_file.write_text(json.dumps({"sig": sig, "complete": True, "n_images": len(paths)}))
    return np.load(cache_file, mmap_mode="r")


def build_bank(
    feats: np.ndarray,
    ratio: float,
    proj_dim: int,
    seed: int,
    min_bank: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Keep `ratio` of the descriptors (at least `min_bank`) as the memory bank.

    The coreset is chosen in a low-dimensional projection for speed, but the bank stores the
    original full-size descriptors.
    """
    n = feats.shape[0]
    bank_size = min(n, max(int(round(n * ratio)), min(n, min_bank)))
    keep = np.sort(coreset_indices(feats, bank_size, proj_dim, seed, device))
    return torch.from_numpy(np.asarray(feats[keep], dtype=np.float32))


@torch.no_grad()
def score_paths(
    model: PatchCore,
    paths: list[Path],
    *,
    batch_size: int,
    resize: int,
    crop: int,
    pre: PreFn | None = None,
    keep_maps: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    scores: list[np.ndarray] = []
    maps: list[np.ndarray] = []
    for i in range(0, len(paths), batch_size):
        x = torch.stack([load_tensor(p, resize, crop, pre) for p in paths[i : i + batch_size]])
        s, m = model.score(x)
        scores.append(s.numpy())
        if keep_maps:
            maps.append(m.numpy())
    return np.concatenate(scores), (np.concatenate(maps) if keep_maps else None)


def train_category(settings: Settings, category: str, force: bool = False) -> dict[str, Any]:
    """Fit the memory bank on the fit split, calibrate thresholds on the calibration split,
    write the artifact. Skips if a valid artifact with the same config hash exists."""
    require_real_weights(settings.backbone.weights)
    configure_runtime(settings)
    root = settings.path("data_dir")
    art_dir = settings.path("artifacts_dir") / category
    ch = settings.config_hash()
    if not force and (art_dir / "manifest.json").exists():
        try:
            _, m = load_artifact(art_dir)
            if m["config_hash"] == ch:
                log.info("%s: artifact up to date, skipping", category)
                return m
        except Exception:  # noqa: BLE001 - rebuild on any problem
            log.warning("%s: existing artifact invalid, rebuilding", category)
    t_start = time.time()
    pc, dc = settings.patchcore, settings.data
    fit, calib = split_fit_calib(
        train_paths(root, category), settings.seed, dc.calib_fraction, dc.min_calib
    )
    extractor = build_extractor(settings)
    cache = settings.path("cache_dir") / category / "fit_feats.npy"
    key = json.dumps(settings.model_config_dict(), sort_keys=True)
    log.info("%s: extracting %d fit images", category, len(fit))
    feats = extract_to_cache(
        extractor, fit, cache, batch_size=pc.batch_size, resize=dc.resize, crop=dc.crop, key=key
    )
    log.info("%s: coreset from %d descriptors", category, feats.shape[0])
    device = settings.resolve_device()
    bank = build_bank(
        feats, pc.coreset_ratio, pc.coreset_proj_dim, settings.seed, pc.min_bank, device
    )
    model = PatchCore(
        extractor,
        bank,
        knn_backend=pc.knn_backend,
        device=device,
        map_sigma=pc.map_sigma,
        crop=dc.crop,
    )
    cal_scores, _ = score_paths(
        model, calib, batch_size=pc.batch_size, resize=dc.resize, crop=dc.crop
    )
    thr = calibrate(cal_scores)
    fields = {
        "category": category,
        "config_hash": ch,
        "seed": settings.seed,
        "backbone": {"name": settings.backbone.name, **extractor.weights_info},
        "model_config": settings.model_config_dict(),
        "dataset_fingerprint": dataset_fingerprint(root, category),
        "thresholds": thr,
        "n_fit": len(fit),
        "n_calib": len(calib),
        "pool_size": int(feats.shape[0]),
        "calib_scores": [float(v) for v in cal_scores],
    }
    manifest = save_artifact(art_dir, bank, fields)
    record_runtime(settings, "train_seconds", category, time.time() - t_start)
    return manifest
