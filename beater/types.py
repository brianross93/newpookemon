"""Core data contracts shared across the agent stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

Buttons = Literal[
    "UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT", "NOOP"
]

ScriptOpKind = Literal["PRESS", "RELEASE", "WAIT"]
PlanletKind = Literal["MENU_SEQUENCE", "NAVIGATE", "INTERACT", "WAIT"]


@dataclass(slots=True)
class Observation:
    """Raw emulator outputs forwarded to perception."""

    rgb: "np.ndarray"  # (160, 144, 3), uint8
    ram: "np.ndarray"  # Raw bytes from emulator RAM.
    step_idx: int
    last_save_slot: Optional[int] = None


@dataclass(slots=True)
class ScriptOp:
    """Executor-ready instruction (button + frames of dwell)."""

    op: ScriptOpKind
    button: Optional[Buttons] = None
    frames: int = 0

    def __post_init__(self) -> None:
        if self.op != "WAIT" and self.button is None:
            raise ValueError("PRESS/RELEASE operations require a button")
        if self.frames < 0:
            raise ValueError("frames must be non-negative")


@dataclass(slots=True)
class Planlet:
    """Composable macro emitted by the plan-head."""

    id: str
    kind: PlanletKind
    args: Dict[str, object]
    script: List[ScriptOp] = field(default_factory=list)
    timeout_steps: int = 0


Script = List[ScriptOp]

__all__ = [
    "Buttons",
    "ScriptOpKind",
    "PlanletKind",
    "Observation",
    "ScriptOp",
    "Planlet",
    "Script",
]
