"""Helpers for lightweight ASCII maps passed to the LLM."""

from __future__ import annotations

from typing import List, Sequence, Tuple

Coord = Tuple[int, int]


def ascii_tile_map(
    tile_grid: List[List[str]],
    anchor: Coord,
    candidates: Sequence[Coord],
    radius: int = 5,
) -> str:
    """Return a small ASCII map centered on the player."""
    if not tile_grid:
        return ""
    h = len(tile_grid)
    w = len(tile_grid[0])
    ar, ac = anchor
    r0 = max(0, ar - radius)
    r1 = min(h, ar + radius + 1)
    c0 = max(0, ac - radius)
    c1 = min(w, ac + radius + 1)
    cand_lookup: dict[Coord, str] = {}
    for idx, coord in enumerate(candidates):
        cand_lookup[coord] = str(idx % 10)
    lines: List[str] = []
    for r in range(r0, r1):
        cells: List[str] = []
        for c in range(c0, c1):
            token = tile_grid[r][c].split(":", 1)[-1]
            cell = token.split("_")[-1][-2:]
            key = (r, c)
            if key == anchor:
                cell = "P "
            elif key in cand_lookup:
                cell = f"{cand_lookup[key]} "
            cells.append(cell)
        lines.append(" ".join(cells))
    return "\n".join(lines)

