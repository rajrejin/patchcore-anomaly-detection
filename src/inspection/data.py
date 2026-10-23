"""MVTec AD dataset layer: listing, splits, verification and image/mask geometry."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

OBJECTS = [
    "bottle", "cable", "capsule", "hazelnut", "metal_nut",
    "pill", "screw", "toothbrush", "transistor", "zipper",
]  # fmt: skip
TEXTURES = ["carpet", "grid", "leather", "tile", "wood"]
CATEGORIES = sorted(OBJECTS + TEXTURES)

EXPECTED_TOTAL = {"train": 3629, "test": 1725, "test_good": 467, "test_defective": 1258}
EXPECTED_PER_CATEGORY = {
    "hazelnut": 501, "screw": 480, "pill": 434, "carpet": 397, "zipper": 391,
    "cable": 374, "leather": 369, "capsule": 351, "tile": 347, "grid": 342,
    "metal_nut": 335, "wood": 326, "transistor": 313, "bottle": 292, "toothbrush": 102,
}  # fmt: skip

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int  # 0 good, 1 defective
    defect_type: str
    mask_path: Path | None


def _pngs(d: Path) -> list[Path]:
    return sorted(d.glob("*.png"))


def train_paths(root: Path, category: str) -> list[Path]:
    return _pngs(root / category / "train" / "good")


def test_samples(root: Path, category: str) -> list[Sample]:
    out: list[Sample] = []
    test_dir = root / category / "test"
    for sub in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        for img in _pngs(sub):
            if sub.name == "good":
                out.append(Sample(img, 0, "good", None))
            else:
                mask = root / category / "ground_truth" / sub.name / f"{img.stem}_mask.png"
                out.append(Sample(img, 1, sub.name, mask))
    return out


def split_fit_calib(
    paths: list[Path], seed: int, calib_fraction: float = 0.1, min_calib: int = 10
) -> tuple[list[Path], list[Path]]:
    """90/10 split with a fixed seed. Calibration gets at least `min_calib` images
    (all of them if fewer exist)."""
    n = len(paths)
    if n <= min_calib:
        return list(paths), list(paths)
    n_cal = max(min_calib, round(n * calib_fraction))
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cal_idx = set(order[:n_cal].tolist())
    fit = [p for i, p in enumerate(paths) if i not in cal_idx]
    cal = [p for i, p in enumerate(paths) if i in cal_idx]
    return fit, cal


def dataset_fingerprint(root: Path, category: str) -> str:
    """Hash of sorted relative paths and file sizes for one category (train and test images)."""
    h = hashlib.sha256()
    base = root / category
    for p in sorted(base.rglob("*.png")):
        h.update(f"{p.relative_to(base).as_posix()}:{p.stat().st_size}\n".encode())
    return h.hexdigest()


# ---------------------------------------------------------------- geometry


def load_rgb(path: Path | str) -> Image.Image:
    return Image.open(path).convert("RGB")


def geometry_image(img: Image.Image, resize: int = 256, crop: int = 224) -> np.ndarray:
    """Resize shorter-side-preserving to a resize x resize square, then center crop.
    Returns uint8 HxWx3 RGB."""
    img = img.resize((resize, resize), Image.Resampling.BILINEAR)
    off = (resize - crop) // 2
    return np.asarray(img.crop((off, off, off + crop, off + crop)), dtype=np.uint8)


def geometry_mask(mask: Image.Image | None, resize: int = 256, crop: int = 224) -> np.ndarray:
    """Same geometry as the image, nearest neighbour. Returns uint8 {0,1} of shape (crop, crop)."""
    if mask is None:
        return np.zeros((crop, crop), dtype=np.uint8)
    m = mask.convert("L").resize((resize, resize), Image.Resampling.NEAREST)
    off = (resize - crop) // 2
    arr = np.asarray(m.crop((off, off, off + crop, off + crop)))
    return (arr > 127).astype(np.uint8)


def normalize(img_u8: np.ndarray) -> torch.Tensor:
    """uint8 HxWx3 RGB -> float32 3xHxW ImageNet-normalised tensor."""
    x = (img_u8.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def load_tensor(
    path: Path | str,
    resize: int = 256,
    crop: int = 224,
    pre: Callable[[np.ndarray], np.ndarray] | None = None,
) -> torch.Tensor:
    arr = geometry_image(load_rgb(path), resize, crop)
    if pre is not None:
        arr = pre(arr)
    return normalize(arr)


class MVTecDataset(Dataset):
    """Items: (tensor 3x224x224, label, mask 224x224 uint8, path string)."""

    def __init__(
        self,
        samples: list[Sample],
        resize: int = 256,
        crop: int = 224,
        pre: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.samples, self.resize, self.crop, self.pre = samples, resize, crop, pre

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, np.ndarray, str]:
        s = self.samples[i]
        arr = geometry_image(load_rgb(s.path), self.resize, self.crop)
        mask = geometry_mask(
            Image.open(s.mask_path) if s.mask_path else None, self.resize, self.crop
        )
        if self.pre is not None:
            arr = self.pre(arr)
        return normalize(arr), s.label, mask, str(s.path)


def paths_to_samples(paths: list[Path]) -> list[Sample]:
    return [Sample(p, 0, "good", None) for p in paths]


# ---------------------------------------------------------------- verification


def verify_dataset(root: Path) -> dict[str, object]:
    """Check the dataset against the documented facts. Never edits the expectations."""
    problems: list[str] = []
    per_cat: dict[str, dict[str, int]] = {}
    for cat in CATEGORIES:
        if not (root / cat).is_dir():
            problems.append(f"missing category directory: {cat}")
            continue
        tr = len(train_paths(root, cat))
        ts = test_samples(root, cat)
        good = sum(1 for s in ts if s.label == 0)
        per_cat[cat] = {
            "train": tr,
            "test": len(ts),
            "test_good": good,
            "test_defective": len(ts) - good,
        }
        total = tr + len(ts)
        if total != EXPECTED_PER_CATEGORY[cat]:
            problems.append(f"{cat}: {total} images, expected {EXPECTED_PER_CATEGORY[cat]}")
        for s in ts:
            if s.label == 1:
                if s.mask_path is None or not s.mask_path.exists():
                    problems.append(f"{cat}: missing mask for {s.path.name}")
                else:
                    with Image.open(s.path) as a, Image.open(s.mask_path) as b:
                        if a.size != b.size:
                            problems.append(f"{cat}: mask size mismatch for {s.path.name}")
    totals = {
        "train": sum(v["train"] for v in per_cat.values()),
        "test": sum(v["test"] for v in per_cat.values()),
        "test_good": sum(v["test_good"] for v in per_cat.values()),
        "test_defective": sum(v["test_defective"] for v in per_cat.values()),
    }
    if len(per_cat) == len(CATEGORIES):
        for k, v in EXPECTED_TOTAL.items():
            if totals[k] != v:
                problems.append(f"total {k}: {totals[k]}, expected {v}")
    return {"ok": not problems, "problems": problems, "per_category": per_cat, "totals": totals}


def available_categories(root: Path) -> list[str]:
    return [c for c in CATEGORIES if (root / c / "test").is_dir() and (root / c / "train").is_dir()]
