"""Metric arithmetic on small hand-computed arrays (not datasets)."""

from __future__ import annotations

import numpy as np
import pytest

from inspection import metrics as M


def test_image_auroc_perfect_and_worst() -> None:
    y = np.array([0, 0, 1, 1])
    assert M.image_auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert M.image_auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_image_auroc_hand_value() -> None:
    # positives 0.35 and 0.8, negatives 0.1 and 0.4: pairs won = 3 of 4
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    assert M.image_auroc(y, s) == pytest.approx(0.75)


def test_bootstrap_ci_deterministic_and_ordered() -> None:
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    s = np.array([0.1, 0.3, 0.2, 0.9, 0.7, 0.4, 0.5, 0.8])
    a = M.bootstrap_auroc_ci(y, s, 200, seed=0)
    b = M.bootstrap_auroc_ci(y, s, 200, seed=0)
    assert a == b
    assert 0.0 <= a[0] <= M.image_auroc(y, s) <= a[1] <= 1.0


def test_bootstrap_ci_one_class_is_nan_not_a_crash() -> None:
    lo, hi = M.bootstrap_auroc_ci(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3]), 20, seed=0)
    assert np.isnan(lo) and np.isnan(hi)


def test_pixel_auroc_toy() -> None:
    masks = np.zeros((2, 2, 2), dtype=np.uint8)
    masks[1, 0, 0] = 1
    maps = np.zeros((2, 2, 2), dtype=np.float32)
    maps[1, 0, 0] = 0.9
    assert M.pixel_auroc(masks, maps) == 1.0


def test_threshold_metrics_hand_counts() -> None:
    labels = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.6, 0.2, 0.7, 0.4])
    m = M.threshold_metrics(labels, scores, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["fpr"] == pytest.approx(1 / 3)


def test_oracle_f1_is_at_least_calibrated() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.5, 0.6, 0.9])
    assert M.oracle_best_f1(labels, scores)["f1"] == 1.0


def test_iou_dice() -> None:
    a = np.array([[1, 1, 0, 0]])
    b = np.array([[0, 1, 1, 0]])
    iou, dice = M.iou_dice(a, b)
    assert iou == pytest.approx(1 / 3)
    assert dice == pytest.approx(0.5)


def test_au_pro_perfect_maps_give_one() -> None:
    masks = np.zeros((1, 10, 10), dtype=np.uint8)
    masks[0, 2:5, 2:5] = 1
    maps = masks.astype(np.float32)
    assert M.au_pro(masks, maps) == pytest.approx(1.0, abs=1e-6)


def test_au_pro_known_partial_overlap() -> None:
    """Hand-computed curve. One region of 4 pixels scoring 1.0, 1.0, 0.5, 0.5. Of the 12 normal
    pixels, two score 0.75 and the rest 0. Sweeping the threshold downwards the curve visits
    (FPR, PRO) = (0, 0.5), (1/6, 0.5), (1/6, 1), (1, 1). The area up to FPR 0.3 is
    (1/6 * 0.5 + (0.3 - 1/6) * 1) / 0.3 = 0.72222."""
    masks = np.zeros((1, 4, 4), dtype=np.uint8)
    masks[0, 0, 0:4] = 1
    maps = np.zeros((1, 4, 4), dtype=np.float32)
    maps[0, 0] = [1.0, 1.0, 0.5, 0.5]
    maps[0, 1, 0:2] = 0.75
    assert M.au_pro(masks, maps) == pytest.approx(0.72222, abs=2e-3)


def test_au_pro_no_regions_is_nan() -> None:
    masks = np.zeros((1, 4, 4), dtype=np.uint8)
    maps = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    assert np.isnan(M.au_pro(masks, maps))
