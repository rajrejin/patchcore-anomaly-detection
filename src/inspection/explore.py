"""Dataset exploration figures: sample grids with mask overlays, counts, mask areas, sizes."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from inspection.config import Settings
from inspection.data import available_categories, test_samples, train_paths

log = logging.getLogger(__name__)


def overlay_mask(
    rgb: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    """Outline a binary mask on an RGB image. Colour is given in RGB order and cv2 is fed BGR."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(bgr, contours, -1, (colour[2], colour[1], colour[0]), 2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def run_exploration(settings: Settings) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = settings.path("data_dir")
    out = settings.path("results_dir") / "figures" / "exploration"
    out.mkdir(parents=True, exist_ok=True)
    cats = available_categories(root)
    stats: dict[str, object] = {}
    defect_counts: dict[str, Counter[str]] = {}
    area_frac: dict[str, list[float]] = {}
    sizes: dict[str, set[tuple[int, int]]] = {}

    ncol = 5
    fig, axes = plt.subplots(len(cats), ncol, figsize=(2.2 * ncol, 2.2 * len(cats)), squeeze=False)
    for r, cat in enumerate(cats):
        tr = train_paths(root, cat)
        samples = test_samples(root, cat)
        defect_counts[cat] = Counter(s.defect_type for s in samples if s.label)
        area_frac[cat] = []
        sizes[cat] = set()
        for s in samples:
            if s.mask_path is not None:
                m = np.asarray(Image.open(s.mask_path).convert("L")) > 127
                area_frac[cat].append(float(m.mean()))
        with Image.open(tr[0]) as im:
            sizes[cat].add(im.size)
        defective = [s for s in samples if s.label][:: max(1, sum(s.label for s in samples) // 3)][
            :3
        ]
        shown = [("train/good", tr[0], None)] + [
            (s.defect_type, s.path, s.mask_path) for s in defective
        ]
        for c in range(ncol):
            ax = axes[r][c]
            ax.set_xticks([]), ax.set_yticks([])
            if c < len(shown):
                title, p, mp = shown[c]
                rgb = np.asarray(Image.open(p).convert("RGB"))
                if mp is not None:
                    rgb = overlay_mask(rgb, np.asarray(Image.open(mp).convert("L")))
                ax.imshow(rgb)
                ax.set_title(f"{cat}: {title}" if c == 0 else title, fontsize=7)
            else:
                ax.axis("off")
        stats[cat] = {
            "n_train": len(tr),
            "n_test": len(samples),
            "defect_types": dict(defect_counts[cat]),
            "image_size": sorted(sizes[cat]),
            "mask_area_fraction_mean": float(np.mean(area_frac[cat])) if area_frac[cat] else None,
        }
    fig.tight_layout()
    fig.savefig(out / "sample_grid.png", dpi=90)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(cats, [sum(defect_counts[c].values()) for c in cats])
    ax.set_ylabel("defective test images")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(out / "defect_counts.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.boxplot([area_frac[c] for c in cats], tick_labels=cats, showfliers=False)
    ax.set_yscale("log")
    ax.set_ylabel("mask area / image area")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(out / "mask_area_fractions.png", dpi=110)
    plt.close(fig)

    (out / "exploration_stats.json").write_text(json.dumps(stats, indent=2))
    return out
