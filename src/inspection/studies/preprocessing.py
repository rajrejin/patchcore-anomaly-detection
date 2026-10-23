"""Image preprocessing variants applied to the 224 x 224 uint8 RGB crop before the backbone."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

PreFn = Callable[[np.ndarray], np.ndarray]


def gaussian_blur(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img, (3, 3), 1.0)


def cielab(img: np.ndarray) -> np.ndarray:
    """RGB to CIELAB (OpenCV 8-bit encoding), then fed through the usual normalisation."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)


def gray_opening(img: np.ndarray) -> np.ndarray:
    """Grayscale morphological opening (3 x 3) replicated to three channels."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return np.repeat(opened[:, :, None], 3, axis=2)


PREPROCESSING: dict[str, PreFn | None] = {
    "baseline": None,
    "gaussian_blur_k3_s1": gaussian_blur,
    "cielab": cielab,
    "gray_opening_3x3": gray_opening,
}
