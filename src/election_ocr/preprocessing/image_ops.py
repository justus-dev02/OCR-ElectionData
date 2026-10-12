"""
Classical image processing (deterministic, no learned models).

All functions operate on numpy arrays in page-image pixel space.
Ink masks are boolean arrays where True = ink (dark pixel).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Global Otsu binarization -> boolean ink mask.

    Otsu is used (instead of adaptive thresholds) because the scans have
    uniform background; adaptive methods create halo artifacts around the
    dense digit columns which then look like extra glyphs.
    """
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary > 0


def estimate_skew(gray: np.ndarray, max_degrees: float) -> Tuple[float, float]:
    """Estimate page skew by maximizing the variance of horizontal ink projections.

    Returns (angle_degrees, sharpness) where sharpness is the relative gain of
    the best angle over the unrotated page (0 = no preference). The angle is
    the rotation to APPLY to deskew the page.
    """
    h, w = gray.shape
    scale = 1200.0 / max(h, w)
    small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else gray
    mask = ink_mask(small).astype(np.uint8)

    def score(angle: float) -> float:
        sh, sw = mask.shape
        m = cv2.getRotationMatrix2D((sw / 2, sh / 2), angle, 1.0)
        rot = cv2.warpAffine(mask, m, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
        return float(np.var(rot.sum(axis=1)))

    base = score(0.0)
    coarse = np.arange(-max_degrees, max_degrees + 1e-9, 0.25)
    best = max(coarse, key=score)
    fine = np.arange(best - 0.25, best + 0.25 + 1e-9, 0.02)
    best_fine = float(max(fine, key=score))
    best_score = score(best_fine)
    sharpness = (best_score - base) / base if base > 0 else 0.0
    return round(best_fine, 3), round(sharpness, 4)


def rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)


@dataclass
class LineMasks:
    text: np.ndarray  # ink without ruling lines
    vertical: np.ndarray  # vertical ruling lines
    horizontal: np.ndarray  # horizontal ruling lines


def separate_ruling_lines(ink: np.ndarray, dpi: int) -> LineMasks:
    """Split an ink mask into text and ruling lines with morphological openings.

    Kernel lengths are derived from the DPI (physical length ~ 6 mm vertical,
    ~ 10 mm horizontal): no glyph of the printed digits is that long.
    """
    ink_u8 = ink.astype(np.uint8) * 255
    v_len = max(15, int(round(dpi * 6 / 25.4)))
    h_len = max(25, int(round(dpi * 10 / 25.4)))
    vertical = cv2.morphologyEx(ink_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len)))
    horizontal = cv2.morphologyEx(ink_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1)))
    # Slightly widen line masks so anti-aliased line borders are removed as well.
    vertical = cv2.dilate(vertical, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
    horizontal = cv2.dilate(horizontal, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    text = (ink_u8 > 0) & ~(vertical > 0) & ~(horizontal > 0)
    return LineMasks(text=text, vertical=vertical > 0, horizontal=horizontal > 0)


def quality_metrics(gray: np.ndarray) -> dict:
    """Simple, reproducible page quality indicators (stored as evidence, not used as gates)."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    ink = ink_mask(gray)
    return {
        "mean_intensity": round(float(gray.mean()), 2),
        "contrast_std": round(float(gray.std()), 2),
        "sharpness_laplacian_var": round(float(lap.var()), 2),
        "ink_ratio": round(float(ink.mean()), 5),
    }
