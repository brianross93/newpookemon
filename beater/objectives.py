"""Objective specification and reward shaping.

LLM proposes a high-level objective spec; the engine computes local rewards
from observable signals using the provided weights and TTL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass(slots=True)
class ObjectiveSpec:
    phase: str = "explore"
    reward_weights: Dict[str, float] = field(default_factory=dict)
    timeouts: Dict[str, int] = field(default_factory=dict)  # e.g., {"ttl_steps": 500}
    skill_bias: Optional[str] = None  # e.g., "menu" or "overworld"


class ObjectiveEngine:
    def __init__(self) -> None:
        self._spec: Optional[ObjectiveSpec] = None
        self._expires_at: Optional[int] = None

    def set_spec(self, spec_dict: Dict[str, object], current_step: int) -> None:
        phase = str(spec_dict.get("phase", "explore"))
        reward_weights = dict(spec_dict.get("reward_weights", {}))
        timeouts = dict(spec_dict.get("timeouts", {}))
        bias = spec_dict.get("skill_bias")
        ttl = int(timeouts.get("ttl_steps", 0) or 0)
        self._spec = ObjectiveSpec(
            phase=phase,
            reward_weights=reward_weights,
            timeouts=timeouts,
            skill_bias=str(bias) if isinstance(bias, str) else None,
        )
        self._expires_at = current_step + ttl if ttl > 0 else None

    def maybe_expired(self, current_step: int) -> bool:
        return self._expires_at is not None and current_step >= self._expires_at

    def skill_bias(self) -> Optional[str]:
        return self._spec.skill_bias if self._spec else None

    def reward(
        self,
        *,
        gate_action: torch.Tensor,
        skill_action: torch.Tensor,
        nav_success: bool,
        sprite_delta: float,
        scene_change: float = 0.0,
        menu_progress: float = 0.0,
        name_committed: float = 0.0,
    ) -> torch.Tensor:
        if not self._spec:
            return torch.tensor([0.0], dtype=torch.float32)
        w = self._spec.reward_weights
        r = 0.0
        if nav_success:
            r += float(w.get("nav_success", 0.0))
        r += float(w.get("sprite_delta", 0.0)) * float(sprite_delta)
        r += float(w.get("scene_change", 0.0)) * float(scene_change)
        r += float(w.get("menu_progress", 0.0)) * float(menu_progress)
        r += float(w.get("name_committed", 0.0)) * float(name_committed)
        return torch.tensor([r], dtype=torch.float32)

    def summary(self) -> str:
        if not self._spec:
            return "objective: <none>"
        ttl = self._spec.timeouts.get("ttl_steps", 0)
        return f"objective: phase={self._spec.phase} bias={self._spec.skill_bias} ttl={ttl} weights={self._spec.reward_weights}"

