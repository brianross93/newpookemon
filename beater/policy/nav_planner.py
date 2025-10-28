"""Breadth-first navigation planner driven by RAM-backed collision data."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

Coord = Tuple[int, int]


@dataclass(slots=True)
class NavPlannerConfig:
    """Configuration for RAM-backed navigation."""

    max_expansions: Optional[int] = None  # Optional cap on BFS expansions.


class NavPlanner:
    """Compute grid paths using BFS over a boolean passability mask."""

    def __init__(self, config: NavPlannerConfig | None = None):
        self.config = config or NavPlannerConfig()

    def plan(self, passable: Sequence[Sequence[bool]], start: Coord, goal: Coord) -> List[Coord]:
        rows = len(passable)
        if rows == 0:
            return []
        cols = len(passable[0])
        if cols == 0:
            return []
        if not self._in_bounds(start, rows, cols) or not self._in_bounds(goal, rows, cols):
            return []
        if not passable[start[0]][start[1]] or not passable[goal[0]][goal[1]]:
            return []

        queue: deque[Coord] = deque([start])
        came_from: dict[Coord, Coord] = {start: start}
        expansions = 0

        while queue:
            current = queue.popleft()
            if current == goal:
                return self._reconstruct_path(came_from, start, goal)
            expansions += 1
            if self.config.max_expansions and expansions > self.config.max_expansions:
                break
            for neighbor in self._neighbors(current):
                r, c = neighbor
                if not self._in_bounds(neighbor, rows, cols):
                    continue
                if not passable[r][c]:
                    continue
                if neighbor in came_from:
                    continue
                came_from[neighbor] = current
                queue.append(neighbor)
        return []

    @staticmethod
    def _in_bounds(coord: Coord, rows: int, cols: int) -> bool:
        r, c = coord
        return 0 <= r < rows and 0 <= c < cols

    @staticmethod
    def _neighbors(coord: Coord) -> Iterable[Coord]:
        r, c = coord
        yield (r + 1, c)
        yield (r - 1, c)
        yield (r, c + 1)
        yield (r, c - 1)

    @staticmethod
    def _reconstruct_path(came_from: dict[Coord, Coord], start: Coord, goal: Coord) -> List[Coord]:
        path: List[Coord] = [goal]
        cur = goal
        while cur != start:
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        return path
