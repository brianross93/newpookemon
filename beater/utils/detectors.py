"""Simple detectors for scene change and menu heuristics."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from beater.types import Planlet


def _hist_from_class_ids(class_ids: torch.Tensor, grid_shape: Tuple[int, int], codebook_size: int = 256) -> np.ndarray:
    """Build a normalized histogram over tile classes for a single sample.

    class_ids: (B, T) or (T,) long tensor. Uses first sample if B>1.
    """
    if class_ids.dim() == 2:
        ids = class_ids[0].detach().cpu().numpy().astype(np.int64)
    else:
        ids = class_ids.detach().cpu().numpy().astype(np.int64)
    # Clamp IDs into [0, codebook_size)
    ids = np.clip(ids, 0, codebook_size - 1)
    hist, _ = np.histogram(ids, bins=codebook_size, range=(0, codebook_size))
    hist = hist.astype(np.float32)
    total = hist.sum() if hist.sum() > 0 else 1.0
    return hist / float(total)


def scene_change_delta(class_ids_before: torch.Tensor, class_ids_after: torch.Tensor, grid_shape: Tuple[int, int]) -> float:
    """Return L1 distance between histograms as a simple scene-change proxy."""
    h0 = _hist_from_class_ids(class_ids_before, grid_shape)
    h1 = _hist_from_class_ids(class_ids_after, grid_shape)
    return float(0.5 * np.abs(h1 - h0).sum())


def infer_menu_flags(planlet: Planlet, scene_delta: float) -> Tuple[float, float]:
    """Heuristic menu progress and commit flags.

    - menu_progress: 1 when we are issuing menu/interact actions and the scene changed a bit.
    - name_committed: 1 if a START press likely confirmed a menu with a noticeable scene change.
    """
    menu_progress = 0.0
    name_committed = 0.0
    if planlet.kind in ("MENU_SEQUENCE", "INTERACT"):
        if scene_delta > 0.05:
            menu_progress = 1.0
        ops = planlet.args.get("ops") if isinstance(planlet.args, dict) else None
        if isinstance(ops, list) and any(str(op).upper() == "START" for op in ops) and scene_delta > 0.15:
            name_committed = 1.0
    return menu_progress, name_committed

