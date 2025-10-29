"""Skill compilation helpers for planlets."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from beater.types import Planlet, PlanletKind, ScriptOp

Coord = Tuple[int, int]

_DIR_TO_BUTTON = {
    (0, 1): "RIGHT",
    (0, -1): "LEFT",
    (1, 0): "DOWN",
    (-1, 0): "UP",
}


@dataclass(slots=True)
class NavPlanletBuilder:
    wait_frames: int = 2

    def from_path(
        self,
        path: Sequence[Coord],
        tile_grid: List[List[str]],
        goal: Coord | None = None,
    ) -> Planlet:
        if len(path) < 2:
            return Planlet(
                id=str(uuid.uuid4()),
                kind="WAIT",
                args={},
                script=[ScriptOp(op="WAIT", frames=self.wait_frames)],
                timeout_steps=self.wait_frames,
            )
        steps = []
        flat_script: List[ScriptOp] = []
        for idx in range(1, len(path)):
            prev = path[idx - 1]
            cur = path[idx]
            delta = (cur[0] - prev[0], cur[1] - prev[1])
            button = _DIR_TO_BUTTON.get(delta)
            if button is None:
                continue
            step_script = [
                ScriptOp(op="PRESS", button=button, frames=3),
                ScriptOp(op="WAIT", frames=self.wait_frames + 1),
                ScriptOp(op="RELEASE", button=button),
                ScriptOp(op="WAIT", frames=1),
            ]
            flat_script.extend(step_script)
            tile_key = tile_grid[cur[0]][cur[1]]
            tile_class = tile_key.split(":", 1)[-1]
            steps.append(
                {
                    "script": step_script,
                    "tile_id": tile_key,
                    "tile_class": tile_class,
                    "target_coord": cur,
                }
            )
        return Planlet(
            id=str(uuid.uuid4()),
            kind="NAVIGATE",
            args={"steps": steps, "path": list(path), "goal": goal or path[-1], "start": path[0]},
            script=flat_script,
            timeout_steps=len(flat_script),
        )
