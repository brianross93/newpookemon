"""Tile-class aware navigation planner with Thompson sampling."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from beater.sr_memory import PassabilityStore

Grid = List[List[str]]
Coord = Tuple[int, int]


@dataclass(slots=True)
class NavPlannerConfig:
    thompson_retries: int = 4
    epsilon: float = 1e-3


class NavPlanner:
    def __init__(self, store: PassabilityStore, config: NavPlannerConfig | None = None):
        self.store = store
        self.config = config or NavPlannerConfig()

    def plan(self, tile_grid: Grid, start: Coord, goal: Coord) -> List[Coord]:
        if not tile_grid:
            return []
        best_path: List[Coord] = []
        best_cost = float("inf")
        for _ in range(max(1, self.config.thompson_retries)):
            costs = self._sample_cost_grid(tile_grid)
            path, cost = self._shortest_path(costs, start, goal)
            if path and cost < best_cost:
                best_path, best_cost = path, cost
        return best_path

    def _sample_cost_grid(self, tile_grid: Grid) -> List[List[float]]:
        cost_grid: List[List[float]] = []
        for row in tile_grid:
            cost_row: List[float] = []
            for key in row:
                tile_class = key.split(":", 1)[-1]
                prob = self.store.sample(tile_class, key)
                prob = max(prob, self.config.epsilon)
                cost_row.append(1.0 / prob)
            cost_grid.append(cost_row)
        return cost_grid

    def _shortest_path(
        self, cost_grid: List[List[float]], start: Coord, goal: Coord
    ) -> Tuple[List[Coord], float]:
        h = len(cost_grid)
        w = len(cost_grid[0])
        in_bounds = lambda r, c: 0 <= r < h and 0 <= c < w
        dist = {(start[0], start[1]): 0.0}
        prev: dict[Coord, Coord] = {}
        heap: List[Tuple[float, Coord]] = [(0.0, start)]

        while heap:
            cur_cost, (r, c) = heapq.heappop(heap)
            if (r, c) == goal:
                return self._reconstruct(prev, start, goal), cur_cost
            if cur_cost > dist[(r, c)]:
                continue
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if not in_bounds(nr, nc):
                    continue
                neigh_cost = cur_cost + cost_grid[nr][nc]
                if neigh_cost < dist.get((nr, nc), float("inf")):
                    dist[(nr, nc)] = neigh_cost
                    prev[(nr, nc)] = (r, c)
                    heapq.heappush(heap, (neigh_cost, (nr, nc)))
        return [], float("inf")

    @staticmethod
    def _reconstruct(prev: dict[Coord, Coord], start: Coord, goal: Coord) -> List[Coord]:
        if goal not in prev and goal != start:
            return []
        path = [goal]
        cur = goal
        while cur != start:
            cur = prev[cur]
            path.append(cur)
        path.reverse()
        return path
