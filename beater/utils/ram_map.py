"""Helpers for reading overworld tile data directly from WRAM."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Set, Tuple

# Dimensions of the visible overworld tilemap (in 8x8 tiles)
SCREEN_WIDTH: int = 20
SCREEN_HEIGHT: int = 18

# Absolute WRAM address of wTileMap in Pokémon Red/Blue.
# wTileMap is a SCREEN_WIDTH x SCREEN_HEIGHT buffer of tile IDs currently on screen.
W_TILE_MAP_ADDR: int = 0xC3A0


def read_tile_map(ram: Sequence[int]) -> List[List[int]]:
    """Extract the onscreen tilemap from WRAM.

    Returns a SCREEN_HEIGHT x SCREEN_WIDTH grid of tile IDs.
    """

    tiles: List[List[int]] = []
    offset = W_TILE_MAP_ADDR
    for _ in range(SCREEN_HEIGHT):
        row = [int(ram[offset + col]) for col in range(SCREEN_WIDTH)]
        tiles.append(row)
        offset += SCREEN_WIDTH
    return tiles


def tokens_from_tile_map(tile_map: Sequence[Sequence[int]]) -> List[List[str]]:
    """Convert numeric tile IDs to human-readable tokens for debugging/ASCII maps."""

    return [[f"tile:{tile:02X}" for tile in row] for row in tile_map]


def passable_mask(
    tile_map: Sequence[Sequence[int]], impassable_tiles: Iterable[int]
) -> List[List[bool]]:
    """Compute a boolean passability grid using the collision table for the current tileset."""

    blocked: Set[int] = {int(t) & 0xFF for t in impassable_tiles}
    grid: List[List[bool]] = []
    for row in tile_map:
        grid.append([tile not in blocked for tile in row])
    return grid


def boundary_exit_candidates(
    passable: Sequence[Sequence[bool]],
    center: Tuple[int, int],
    limit: int = 8,
) -> List[Tuple[int, int]]:
    """Return passable tiles near boundaries that look like exits or stairs.

    Heuristic: prefer tiles that touch the visible boundary or are adjacent to
    blocked tiles (walls/stairs) and are close to the player.
    """

    rows = len(passable)
    if rows == 0:
        return []
    cols = len(passable[0])
    if cols == 0:
        return []
    cr, cc = center
    scored: List[Tuple[Tuple[int, int, int], Tuple[int, int]]] = []
    for r in range(rows):
        for c in range(cols):
            if not passable[r][c]:
                continue
            touches_edge = r == 0 or r == rows - 1 or c == 0 or c == cols - 1
            blocked_neighbors = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    touches_edge = True
                    continue
                if not passable[nr][nc]:
                    blocked_neighbors += 1
            if not touches_edge and blocked_neighbors == 0:
                continue
            dist = abs(r - cr) + abs(c - cc)
            priority = (0 if touches_edge else 1, -blocked_neighbors, dist)
            scored.append((priority, (r, c)))
    scored.sort(key=lambda item: item[0])
    seen: Set[Tuple[int, int]] = set()
    ordered: List[Tuple[int, int]] = []
    for _, coord in scored:
        if coord in seen:
            continue
        seen.add(coord)
        ordered.append(coord)
        if len(ordered) >= limit:
            break
    return ordered


# --------------------------------------------------------------------------- items
MAX_BAG_SLOTS: int = 20
BAG_COUNT_ADDR: int = 0xD31D
BAG_ITEMS_ADDR: int = 0xD31E


def decode_bag_items(ram: Sequence[int], max_slots: int = MAX_BAG_SLOTS) -> List[dict]:
    """Return a compact representation of the player's current bag contents."""

    if ram is None or len(ram) <= BAG_ITEMS_ADDR:
        return []
    total_reported = int(ram[BAG_COUNT_ADDR])
    items: List[dict] = []
    for slot in range(max_slots):
        base = BAG_ITEMS_ADDR + slot * 2
        if base + 1 >= len(ram):
            break
        item_id = int(ram[base])
        quantity = int(ram[base + 1])
        if item_id in (0x00, 0xFF) or quantity == 0:
            continue
        items.append(
            {
                "slot": slot + 1,
                "item_id": item_id,
                "item_hex": f"0x{item_id:02X}",
                "quantity": quantity,
            }
        )
        if total_reported and len(items) >= total_reported:
            break
    return items


# --------------------------------------------------------------------------- party
PARTY_COUNT_ADDR: int = 0xD163
PARTY_LIST_ADDR: int = 0xD164
PARTY_MON_BASE: int = 0xD16B
PARTY_MON_STRIDE: int = 0x2C
PARTY_CUR_HP_OFFSET: int = 1
PARTY_STATUS_OFFSET: int = 4
PARTY_MOVES_OFFSET: int = 8
PARTY_ACTUAL_LEVEL_OFFSET: int = 0x21
PARTY_MAX_HP_OFFSET: int = 0x22


def _read_u16_le(ram: Sequence[int], offset: int) -> int:
    lo = int(ram[offset])
    hi = int(ram[offset + 1]) if offset + 1 < len(ram) else 0
    return lo | (hi << 8)


def decode_party(ram: Sequence[int]) -> List[dict]:
    """Return a list summarising the current party (species ids, HP, level, moves)."""

    if ram is None or len(ram) <= PARTY_LIST_ADDR:
        return []
    count = int(ram[PARTY_COUNT_ADDR])
    party: List[dict] = []
    for idx in range(min(count, 6)):
        species = int(ram[PARTY_LIST_ADDR + idx])
        if species in (0x00, 0xFF):
            continue
        base = PARTY_MON_BASE + idx * PARTY_MON_STRIDE
        if base + PARTY_MON_STRIDE > len(ram):
            break
        current_hp = _read_u16_le(ram, base + PARTY_CUR_HP_OFFSET)
        max_hp = _read_u16_le(ram, base + PARTY_MAX_HP_OFFSET)
        level = int(ram[base + PARTY_ACTUAL_LEVEL_OFFSET])
        status = int(ram[base + PARTY_STATUS_OFFSET])
        moves = [
            int(ram[base + PARTY_MOVES_OFFSET + mv])
            for mv in range(4)
            if base + PARTY_MOVES_OFFSET + mv < len(ram)
        ]
        party.append(
            {
                "slot": idx + 1,
                "species_id": species,
                "species_hex": f"0x{species:02X}",
                "level": level,
                "hp": current_hp,
                "max_hp": max_hp,
                "hp_percent": (current_hp / max_hp) if max_hp else None,
                "status": status,
                "moves": moves,
            }
        )
    return party
