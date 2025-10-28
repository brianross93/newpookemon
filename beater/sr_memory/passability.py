"""Discovery-first passability store (Beta priors for tiles)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(slots=True)
class BetaPosterior:
    alpha: float = 1.0
    beta: float = 1.0

    def update(self, success: bool, weight: float = 1.0) -> None:
        if success:
            self.alpha += weight
        else:
            self.beta += weight

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / ((total ** 2) * (total + 1))

    def sample(self, rng: np.random.Generator) -> float:
        return rng.beta(self.alpha, self.beta)


@dataclass(slots=True)
class PassabilityEstimate:
    tile_class: str
    tile_id: str
    class_mean: float
    instance_mean: float
    class_variance: float
    instance_variance: float
    blended: float


class PassabilityStore:
    """Maintains class-level and instance-level Beta posteriors."""

    def __init__(self, class_weight: float = 0.5, seed: Optional[int] = None):
        self.class_posteriors: Dict[str, BetaPosterior] = {}
        self.instance_posteriors: Dict[str, BetaPosterior] = {}
        self.class_weight = class_weight
        self.rng = np.random.default_rng(seed)

    def update(
        self,
        tile_class: str,
        tile_id: str,
        success: bool,
        *,
        weight: float = 1.0,
    ) -> PassabilityEstimate:
        cls_post = self.class_posteriors.setdefault(tile_class, BetaPosterior())
        inst_post = self.instance_posteriors.setdefault(tile_id, BetaPosterior())
        cls_post.update(success, weight)
        inst_post.update(success, weight)
        return self.get_estimate(tile_class, tile_id)

    def get_estimate(self, tile_class: str, tile_id: str) -> PassabilityEstimate:
        cls_post = self.class_posteriors.setdefault(tile_class, BetaPosterior())
        inst_post = self.instance_posteriors.setdefault(tile_id, BetaPosterior())
        blended = self.class_weight * cls_post.mean + (1.0 - self.class_weight) * inst_post.mean
        blended = max(0.2, blended)
        return PassabilityEstimate(
            tile_class=tile_class,
            tile_id=tile_id,
            class_mean=cls_post.mean,
            instance_mean=inst_post.mean,
            class_variance=cls_post.variance,
            instance_variance=inst_post.variance,
            blended=blended,
        )

    def sample(self, tile_class: str, tile_id: str) -> float:
        """Thompson sample for navigation."""
        cls_post = self.class_posteriors.setdefault(tile_class, BetaPosterior())
        inst_post = self.instance_posteriors.setdefault(tile_id, BetaPosterior())
        cls_samp = cls_post.sample(self.rng)
        inst_samp = inst_post.sample(self.rng)
        return self.class_weight * cls_samp + (1.0 - self.class_weight) * inst_samp

    def get_class_mean(self, tile_class: str) -> float:
        return self.class_posteriors.setdefault(tile_class, BetaPosterior()).mean

    def get_instance_mean(self, tile_id: str) -> float:
        return self.instance_posteriors.setdefault(tile_id, BetaPosterior()).mean
