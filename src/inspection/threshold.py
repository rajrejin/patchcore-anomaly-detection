"""Label-free threshold calibration on held-out normal images."""

from __future__ import annotations

import numpy as np


def calibrate(calib_scores: np.ndarray) -> dict[str, float | int]:
    """Image threshold = maximum image score over calibration normals (conservative,
    label-free). The pixel threshold is the same value. Test labels are never used."""
    s = np.asarray(calib_scores, dtype=np.float64)
    if s.size == 0:
        raise ValueError("no calibration scores")
    t = float(s.max())
    return {
        "image": t,
        "pixel": t,
        "n_calib": int(s.size),
        "calib_mean": float(s.mean()),
        "calib_std": float(s.std()),
    }


def apply_alpha(pixel_threshold: float, alpha: float) -> float:
    """Sensitivity analysis only; never used to set the deployed threshold."""
    return pixel_threshold * alpha
