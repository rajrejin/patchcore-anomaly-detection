"""Evaluation metrics: image AUROC (+ bootstrap CI), pixel AUROC, AU-PRO, threshold metrics."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score

EIGHT_CONN = np.ones((3, 3), dtype=int)


def image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def bootstrap_auroc_ci(
    labels: np.ndarray, scores: np.ndarray, n_boot: int = 1000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap CI over images. Resamples with a single class are skipped."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    rng = np.random.default_rng(seed)
    n = len(labels)
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        lb = labels[idx]
        if lb.min() == lb.max():
            continue
        vals.append(roc_auc_score(lb, scores[idx]))
    if not vals:  # every resample had one class only (tiny or one-class input)
        return float("nan"), float("nan")
    a = (1 - level) / 2
    lo, hi = np.quantile(vals, [a, 1 - a])
    return float(lo), float(hi)


def pixel_auroc(masks: np.ndarray, maps: np.ndarray) -> float:
    """AUROC over all pixels of all images (normal images carry all-zero masks)."""
    return float(roc_auc_score(masks.reshape(-1).astype(np.uint8), maps.reshape(-1)))


def au_pro(
    masks: np.ndarray, maps: np.ndarray, max_fpr: float = 0.3, n_thresholds: int = 200
) -> float:
    """Area under the per-region overlap (PRO) curve up to `max_fpr`, normalised by `max_fpr`.

    PRO at a threshold is the mean, over 8-connected ground-truth regions, of the fraction of
    the region's pixels scoring above the threshold. FPR is measured on all pixels outside the
    ground-truth masks. Thresholds are `n_thresholds` evenly spaced values over the score range.
    Re-implementation at the evaluation resolution; may differ slightly from official code.
    """
    masks = masks.astype(bool)
    lo, hi = float(maps.min()), float(maps.max())
    if hi <= lo:
        return 0.0
    # one extra threshold below every score so the curve reaches (FPR 1, PRO 1)
    edges = np.concatenate([[lo - 1e-6 * (hi - lo)], np.linspace(lo, hi, n_thresholds)])
    n_thresholds = len(edges) - 1

    def exceed_counts(values: np.ndarray) -> np.ndarray:
        """Number of values strictly above each threshold in `edges`."""
        s = np.sort(values.reshape(-1))
        return (s.size - np.searchsorted(s, edges, side="right")).astype(np.float64)

    neg = maps[~masks]
    fpr = exceed_counts(neg) / max(neg.size, 1)

    pro_sum = np.zeros(n_thresholds + 1)
    n_regions = 0
    for mask_i, map_i in zip(masks, maps, strict=True):
        if not mask_i.any():
            continue
        labeled, n = ndimage.label(mask_i, structure=EIGHT_CONN)
        for r in range(1, n + 1):
            region = map_i[labeled == r]
            pro_sum += exceed_counts(region) / region.size
            n_regions += 1
    if n_regions == 0:
        return float("nan")
    pro = pro_sum / n_regions

    order = np.lexsort((pro, fpr))  # by FPR, ties ordered by PRO so the upper envelope is last
    fpr, pro = fpr[order], pro[order]
    keep = fpr <= max_fpr
    f, p = fpr[keep], pro[keep]
    if keep.sum() < len(fpr):  # interpolate the curve at max_fpr
        j = int(keep.sum())
        f0, f1, p0, p1 = fpr[j - 1], fpr[j], pro[j - 1], pro[j]
        pm = p0 + (p1 - p0) * (max_fpr - f0) / (f1 - f0) if f1 > f0 else p0
        f, p = np.append(f, max_fpr), np.append(p, pm)
    area = float(np.sum((p[1:] + p[:-1]) / 2.0 * np.diff(f)))  # trapezoid rule, any numpy
    return area / max_fpr


def threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Precision, recall, F1, FPR (on normals); anomalous iff score > threshold."""
    labels = np.asarray(labels).astype(bool)
    pred = np.asarray(scores) > threshold
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    tn = int((~pred & ~labels).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def oracle_best_f1(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """ORACLE: best F1 over all thresholds, chosen with test labels. Analysis only."""
    best = {"f1": -1.0, "threshold": float("nan")}
    cand = np.unique(scores)
    mids = np.concatenate([[cand[0] - 1e-9], (cand[:-1] + cand[1:]) / 2, [cand[-1] + 1e-9]])
    for t in mids:
        m = threshold_metrics(labels, scores, float(t))
        if m["f1"] > best["f1"]:
            best = {
                "f1": m["f1"],
                "threshold": float(t),
                "precision": m["precision"],
                "recall": m["recall"],
            }
    return best


def iou_dice(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    p, g = pred.astype(bool), gt.astype(bool)
    inter = float((p & g).sum())
    union = float((p | g).sum())
    iou = inter / union if union else 1.0
    denom = float(p.sum() + g.sum())
    dice = 2 * inter / denom if denom else 1.0
    return iou, dice
