"""Waypoint/goal management for dynamic navigation."""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

Coord = Tuple[int, int]


class GoalManager:
    def __init__(self):
        self._queue: Deque[Coord] = deque()
        self._score: Dict[Coord, float] = {}
        self._fail_count: Dict[Coord, int] = {}

    def peek_candidates(self, grid_shape: Tuple[int, int], count: int = 4) -> List[Coord]:
        """Return up to `count` best candidates by priority (not just queue order)."""
        self._ensure_seed(grid_shape)
        center = self.player_anchor(grid_shape)
        ranked = sorted(
            list(self._queue), key=lambda c: self._priority(c, center), reverse=True
        )
        return ranked[: min(count, len(ranked))]

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
            # Choose by distance with penalties for repeated failures and low score.
            goal = max(self._queue, key=lambda c: self._priority(c, center))
            self._queue.remove(goal)
        else:
            goal = self._queue.popleft()
        # Re-enqueue at back to avoid immediate repeats regardless of outcome.
        self._queue.append(goal)
        return goal

    def feedback(self, goal: Coord, success: bool) -> None:
        """Update internal scores and cool-down failed goals.

        - Success: increase score and reset fail count.
        - Failure: decrease score, increment fail count, and keep the goal at the back
          of the queue so it won't be retried immediately.
        """
        delta = 1.0 if success else -0.5
        self._score[goal] = self._score.get(goal, 0.0) + delta
        if success:
            self._fail_count[goal] = 0
        else:
            self._fail_count[goal] = self._fail_count.get(goal, 0) + 1
            # Move to the back (if present) to avoid front-of-queue retries.
            try:
                self._queue.remove(goal)
            except ValueError:
                pass
            self._queue.append(goal)

    def player_anchor(self, grid_shape: Tuple[int, int]) -> Coord:
        return (grid_shape[0] // 2, grid_shape[1] // 2)

    def unseen_candidates(
        self,
        grid_shape: Tuple[int, int],
        tile_grid: List[List[str]],
        pass_store,
        count: int = 4,
    ) -> List[Coord]:
        """Return up to `count` coordinates that remain largely unexplored."""

        center = self.player_anchor(grid_shape)
        cr, cc = center
        radius = 4
        unseen: List[Coord] = []
        r0 = max(0, cr - radius)
        r1 = min(len(tile_grid) - 1, cr + radius)
        c0 = max(0, cc - radius)
        c1 = min(len(tile_grid[0]) - 1, cc + radius)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                key = tile_grid[r][c]
                cls = key.split(":", 1)[-1]
                est = pass_store.get_estimate(cls, key)
                if abs(est.instance_mean - 0.5) < 0.05:
                    unseen.append((r, c))
        unseen.sort(key=lambda coord: self._distance(coord, center))
        dedup: List[Coord] = []
        for coord in unseen:
            if coord not in dedup:
                dedup.append(coord)
            if len(dedup) >= count:
                break
        return dedup

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

    def _priority(self, coord: Coord, center: Coord) -> float:
        """Higher is better. Distance minus failure penalty plus learned score.

        The failure penalty grows sublinearly to allow occasional retries while
        still diversifying exploration when a boundary is repeatedly hit.
        """
        dist = self._distance(coord, center)
        score = self._score.get(coord, 0.0)
        fail = self._fail_count.get(coord, 0)
        penalty = math.sqrt(float(fail)) * 1.0
        return dist + score - penalty

    # Public helpers -----------------------------------------------------------
    def get_fail_count(self, coord: Coord) -> int:
        return int(self._fail_count.get(coord, 0))
