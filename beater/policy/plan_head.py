"""Plan-head that decodes skills into Planlets."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List

import torch

from beater.types import Planlet, PlanletKind, ScriptOp


@dataclass(slots=True)
class PlanHeadConfig:
    skill_vocab: List[PlanletKind]
    default_timeout: int = 32


class PlanHead:
    def __init__(self, config: PlanHeadConfig):
        self.config = config

    def decode(
        self,
        skill_logits: torch.Tensor,
        timeout: torch.Tensor,
        skill_action: torch.Tensor | None = None,
        preferred_kind: PlanletKind | None = None,
    ) -> Planlet:
        if skill_action is not None:
            skill_idx = int(skill_action.item())
        else:
            skill_idx = int(torch.argmax(skill_logits, dim=-1).item())
        skill_idx = skill_idx % len(self.config.skill_vocab)
        plan_kind = self.config.skill_vocab[skill_idx]
        if preferred_kind in self.config.skill_vocab and plan_kind == "WAIT":
            plan_kind = preferred_kind  # allow options to override idle planlets
        timeout_steps = int(timeout.item()) if timeout is not None else self.config.default_timeout
        script = self._build_script(plan_kind, timeout_steps)
        return Planlet(
            id=str(uuid.uuid4()),
            kind=plan_kind,
            args={"skill_idx": skill_idx},
            script=script,
            timeout_steps=timeout_steps,
        )

    def _build_script(self, kind: PlanletKind, timeout: int) -> List[ScriptOp]:
        if kind == "NAVIGATE":
            return [
                ScriptOp(op="PRESS", button="UP", frames=2),
                ScriptOp(op="WAIT", frames=timeout),
                ScriptOp(op="RELEASE", button="UP"),
            ]
        if kind == "INTERACT":
            return [
                ScriptOp(op="PRESS", button="A", frames=2),
                ScriptOp(op="WAIT", frames=timeout),
                ScriptOp(op="RELEASE", button="A"),
            ]
        if kind == "MENU_SEQUENCE":
            return [
                ScriptOp(op="PRESS", button="START", frames=2),
                ScriptOp(op="WAIT", frames=timeout),
                ScriptOp(op="RELEASE", button="START"),
            ]
        # WAIT fallback
        return [ScriptOp(op="WAIT", frames=timeout)]
