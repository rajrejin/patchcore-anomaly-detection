"""OpenCV post-processing of anomaly maps: mask cleanup, boxes, rendering, evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from inspection.config import Settings
from inspection.data import geometry_image, load_rgb, test_samples
from inspection.metrics import iou_dice

log = logging.getLogger(__name__)

Box = tuple[int, int, int, int, float]  # x, y, w, h, score


def binarize(amap: np.ndarray, threshold: float) -> np.ndarray:
    return (amap > threshold).astype(np.uint8)


def clean_mask(mask: np.ndarray, kernel: int = 5) -> np.ndarray:
    """Morphological opening then closing on the anomaly mask (not on the input image)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)


def find_boxes(
    mask: np.ndarray, amap: np.ndarray, min_area: int
) -> tuple[list[Box], list[np.ndarray]]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    kept = [c for c in contours if cv2.contourArea(c) >= min_area]
    boxes: list[Box] = []
    for c in kept:
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((int(x), int(y), int(w), int(h), float(amap[y : y + h, x : x + w].max())))
    return boxes, kept


def process_map(amap: np.ndarray, threshold: float, kernel: int, min_area: int) -> dict[str, Any]:
    raw = binarize(amap, threshold)
    cleaned = clean_mask(raw, kernel)
    boxes, contours = find_boxes(cleaned, amap, min_area)
    return {"raw": raw, "clean": cleaned, "boxes": boxes, "contours": contours}


def heatmap_overlay(
    rgb: np.ndarray, amap: np.ndarray, threshold: float, alpha: float = 0.5
) -> np.ndarray:
    """RGB uint8 image blended with a colormapped map / (2 * threshold), clipped to [0, 1]."""
    norm = np.clip(amap / (2.0 * threshold), 0.0, 1.0)
    heat_bgr = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1 - alpha, heat_rgb, alpha, 0)


def annotate(
    rgb: np.ndarray,
    boxes: list[Box],
    score: float,
    contours: list[np.ndarray] | None = None,
    anomalous: bool = True,
) -> np.ndarray:
    """Draw boxes, optional contours and a score label. Input and output are RGB."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if contours:
        cv2.drawContours(bgr, contours, -1, (0, 255, 255), 1)
    for x, y, w, h, _ in boxes:
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)
    colour = (0, 0, 255) if anomalous else (0, 160, 0)
    label = f"{'ANOMALY' if anomalous else 'OK'} {score:.2f}"
    cv2.putText(bgr, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(bgr, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def gt_region_boxes(gt: np.ndarray) -> list[tuple[int, int, int, int]]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), connectivity=8)
    return [(int(s[0]), int(s[1]), int(s[2]), int(s[3])) for s in stats[1:n]]


def evaluate_postprocess(settings: Settings, category: str) -> dict[str, Any]:
    from inspection.evaluate import load_cached_test

    pp = settings.postprocess
    d = load_cached_test(settings, category)
    maps, masks, scores, labels = (
        d["test_maps"],
        d["test_masks"],
        d["test_scores"],
        d["test_labels"],
    )
    res_file = settings.path("results_dir") / "categories" / f"{category}.json"
    base = json.loads(res_file.read_text())
    t_img, t_pix = base["threshold_image"], base["threshold_pixel"]
    flagged = scores > t_img

    def per_image(idx: np.ndarray, thr: float) -> dict[str, float]:
        raw_i, raw_d, cl_i, cl_d = [], [], [], []
        hits_img, box_hits, n_boxes = 0, 0, 0
        for i in idx:
            r = process_map(maps[i], thr, pp.morph_kernel, pp.min_area)
            a, b = iou_dice(r["raw"], masks[i])
            c, e = iou_dice(r["clean"], masks[i])
            raw_i.append(a)
            raw_d.append(b)
            cl_i.append(c)
            cl_d.append(e)
            gts = gt_region_boxes(masks[i])
            ok = [any(box_iou(bx[:4], g) >= pp.box_iou_hit for g in gts) for bx in r["boxes"]]
            hits_img += any(ok)
            box_hits += sum(ok)
            n_boxes += len(ok)
        n = max(len(idx), 1)
        return {
            "n_images": int(len(idx)),
            "raw_mean_iou": float(np.mean(raw_i)) if raw_i else float("nan"),
            "raw_mean_dice": float(np.mean(raw_d)) if raw_d else float("nan"),
            "clean_mean_iou": float(np.mean(cl_i)) if cl_i else float("nan"),
            "clean_mean_dice": float(np.mean(cl_d)) if cl_d else float("nan"),
            "box_hit_rate_images": hits_img / n,
            "box_precision": box_hits / n_boxes if n_boxes else float("nan"),
            "boxes_per_image": n_boxes / n,
        }

    defective = np.where(labels == 1)[0]
    flagged_def = np.where((labels == 1) & flagged)[0]
    out: dict[str, Any] = {
        "category": category,
        "seed": settings.seed,
        "box_hit_definition": (
            f"a predicted box counts as a hit when its IoU with the bounding rectangle of a "
            f"ground truth region is at least {pp.box_iou_hit}"
        ),
        "threshold_pixel": t_pix,
        "flagged_defective": per_image(flagged_def, t_pix),
        "all_defective": per_image(defective, t_pix),
    }
    # alpha sensitivity (analysis only): thresholded map quality and false positive pixels
    normal = np.where(labels == 0)[0]
    sens = []
    for alpha in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2):
        thr = t_pix * alpha
        ious = [iou_dice(maps[i] > thr, masks[i])[0] for i in defective]
        dice = [iou_dice(maps[i] > thr, masks[i])[1] for i in defective]
        fp_frac = float(np.mean([(maps[i] > thr).mean() for i in normal])) if len(normal) else 0.0
        normal_flagged = (
            float(np.mean([(maps[i] > thr).any() for i in normal])) if len(normal) else 0.0
        )
        sens.append(
            {
                "alpha": alpha,
                "mean_iou_defective": float(np.mean(ious)),
                "mean_dice_defective": float(np.mean(dice)),
                "normal_pixel_fp_fraction": fp_frac,
                "normal_images_with_any_pixel": normal_flagged,
            }
        )
    out["alpha_sensitivity"] = sens
    pdir = settings.path("results_dir") / "postprocess"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / f"{category}.json").write_text(json.dumps(out, indent=2))
    render_examples(settings, category, maps, masks, scores, labels, t_img, t_pix)
    return out


def render_examples(
    settings: Settings,
    category: str,
    maps: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    t_img: float,
    t_pix: float,
    n: int = 3,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pp = settings.postprocess
    samples = test_samples(settings.path("data_dir"), category)
    defect_idx = [i for i in np.where(labels == 1)[0] if scores[i] > t_img]
    pick = defect_idx[:: max(1, len(defect_idx) // n)][:n] if defect_idx else []
    titles = ["original", "ground truth", "raw map", "thresholded", "cleaned", "annotated"]
    fig, axes = plt.subplots(
        max(len(pick), 1), 6, figsize=(15, 2.6 * max(len(pick), 1)), squeeze=False
    )
    for r, i in enumerate(pick):
        rgb = geometry_image(load_rgb(samples[i].path), settings.data.resize, settings.data.crop)
        res = process_map(maps[i], t_pix, pp.morph_kernel, pp.min_area)
        panels = [
            rgb,
            masks[i] * 255,
            maps[i],
            res["raw"] * 255,
            res["clean"] * 255,
            annotate(
                heatmap_overlay(rgb, maps[i], t_pix),
                res["boxes"],
                float(scores[i]),
                res["contours"],
            ),
        ]
        for c, p in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(p, cmap=None if p.ndim == 3 else ("inferno" if c == 2 else "gray"))
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=9)
    fig.tight_layout()
    out = settings.path("results_dir") / "figures" / "postprocess" / f"{category}_examples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
