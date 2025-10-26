"""Waypoint/goal management for dynamic navigation."""

from __future__ import annotations

import math
import random
from collections import deque
from itertools import islice
from typing import Deque, Dict, List, Optional, Tuple

Coord = Tuple[int, int]


class GoalManager:
    def __init__(self):
        self._queue: Deque[Coord] = deque()
        self._score: Dict[Coord, float] = {}

    def peek_candidates(self, grid_shape: Tuple[int, int], count: int = 4) -> List[Coord]:
        self._ensure_seed(grid_shape)
        return list(islice(self._queue, 0, min(count, len(self._queue))))

    def next_goal(
        self,
        grid_shape: Tuple[int, int],
        preferred: Optional[str] = None,
        goal_override: Optional[Coord] = None,
    ) -> Coord:
        self._ensure_seed(grid_shape)
        center = self.player_anchor(grid_shape)
        if goal_override and goal_override in self._queue:
            self._queue.remove(goal_override)
            goal = goal_override
        elif preferred == "NAVIGATE":
            goal = max(self._queue, key=lambda c: self._distance(c, center))
            self._queue.remove(goal)
        else:
            goal = self._queue.popleft()
        self._queue.append(goal)
        return goal

    def feedback(self, goal: Coord, success: bool) -> None:
        delta = 1.0 if success else -0.5
        self._score[goal] = self._score.get(goal, 0.0) + delta
        # Reinsert goal near front if it keeps failing.
        if not success:
            try:
                self._queue.remove(goal)
            except ValueError:
                pass
            self._queue.appendleft(goal)

    def player_anchor(self, grid_shape: Tuple[int, int]) -> Coord:
        return (grid_shape[0] // 2, grid_shape[1] // 2)

    # ------------------------------------------------------------------ helpers
    def _ensure_seed(self, grid_shape: Tuple[int, int]) -> None:
        if not self._queue:
            self._seed(grid_shape)

    def _seed(self, grid_shape: Tuple[int, int]) -> None:
        h, w = grid_shape
        candidates = [
            (max(0, h // 2 - 2), w // 2),
            (min(h - 1, h // 2 + 2), w // 2),
            (h // 2, max(0, w // 2 - 3)),
            (h // 2, min(w - 1, w // 2 + 3)),
            (0, w // 2),
            (h - 1, w // 2),
            (h // 2, 0),
            (h // 2, w - 1),
        ]
        random.shuffle(candidates)
        self._queue.extend(candidates)

    @staticmethod
    def _distance(a: Coord, b: Coord) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])
