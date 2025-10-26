"""Sprite/pose tracker used to judge movement success."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from beater.types import Observation


@dataclass(slots=True)
class SpritePose:
    y: float
    x: float
    strength: float


class SpriteMovementDetector:
    """Estimates sprite centroids in a central ROI to detect movement."""

    def __init__(
        self,
        diff_threshold: float = 12.0,
        min_pixels: int = 40,
        roi_ratio: float = 0.6,
        success_px: float = 1.5,
    ):
        self.diff_threshold = diff_threshold
        self.min_pixels = min_pixels
        self.roi_ratio = roi_ratio
        self.success_px = success_px

    def evaluate(self, prev: Observation, new: Observation) -> tuple[bool, float]:
        pose_prev = self._estimate_pose(prev.rgb)
        pose_new = self._estimate_pose(new.rgb)
        if pose_prev is None or pose_new is None:
            return False, 0.0
        dy = pose_new.y - pose_prev.y
        dx = pose_new.x - pose_prev.x
        dist = float(np.hypot(dy, dx))
        return dist >= self.success_px, dist

    # ------------------------------------------------------------------ helpers
    def _estimate_pose(self, frame: np.ndarray) -> Optional[SpritePose]:
        gray = frame.mean(axis=2)
        h, w = gray.shape
        roi_h = int(h * self.roi_ratio)
        roi_w = int(w * self.roi_ratio)
        y0 = (h - roi_h) // 2
        x0 = (w - roi_w) // 2
        roi = gray[y0 : y0 + roi_h, x0 : x0 + roi_w]
        mean = roi.mean()
        mask = np.abs(roi - mean) > self.diff_threshold
        count = mask.sum()
        if count < self.min_pixels:
            return None
        ys, xs = np.nonzero(mask)
        weights = np.abs(roi[mask] - mean)
        strength = float(weights.sum())
        if strength <= 0:
            return None
        y = float((ys * weights).sum() / strength) + y0
        x = float((xs * weights).sum() / strength) + x0
        return SpritePose(y=y, x=x, strength=strength)
