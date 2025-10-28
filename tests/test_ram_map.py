import numpy as np

from beater.utils.ram_map import (
    BAG_COUNT_ADDR,
    BAG_ITEMS_ADDR,
    PARTY_ACTUAL_LEVEL_OFFSET,
    PARTY_COUNT_ADDR,
    PARTY_CUR_HP_OFFSET,
    PARTY_LIST_ADDR,
    PARTY_MAX_HP_OFFSET,
    PARTY_MON_BASE,
    PARTY_STATUS_OFFSET,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    W_TILE_MAP_ADDR,
    decode_bag_items,
    decode_party,
    passable_mask,
    read_tile_map,
    tokens_from_tile_map,
)


def _build_ram_with_tiles(values: list[int]) -> np.ndarray:
    """Helper to write the provided tile ids into consecutive wTileMap bytes."""

    ram = np.zeros(0x10000, dtype=np.uint8)
    total = SCREEN_HEIGHT * SCREEN_WIDTH
    flattened = (values * ((total + len(values) - 1) // len(values)))[:total]
    for idx, tile in enumerate(flattened):
        ram[W_TILE_MAP_ADDR + idx] = tile & 0xFF
    return ram


def test_read_tile_map_returns_expected_dimensions() -> None:
    ram = _build_ram_with_tiles(list(range(16)))
    tile_map = read_tile_map(ram)
    assert len(tile_map) == SCREEN_HEIGHT
    assert len(tile_map[0]) == SCREEN_WIDTH
    assert tile_map[0][0] == 0
    assert tile_map[0][1] == 1
    # Ensure wrapping from the helper works
    assert tile_map[-1][-1] == (SCREEN_HEIGHT * SCREEN_WIDTH - 1) % 16


def test_tokens_and_passable_mask() -> None:
    ram = _build_ram_with_tiles([0x01, 0x02, 0x03, 0x04])
    tile_map = read_tile_map(ram)
    tokens = tokens_from_tile_map(tile_map)
    assert tokens[0][0] == "tile:01"
    # Mark 0x02 and 0x04 as impassable
    mask = passable_mask(tile_map, impassable_tiles={0x02, 0x04})
    assert mask[0][0] is True  # 0x01
    assert mask[0][1] is False  # 0x02
    assert mask[1][0] is True  # 0x03
    assert mask[1][1] is False  # 0x04


def test_decode_bag_items_respects_count() -> None:
    ram = np.zeros((0x10000,), dtype=np.uint8)
    ram[BAG_COUNT_ADDR] = 2
    ram[BAG_ITEMS_ADDR] = 0x01
    ram[BAG_ITEMS_ADDR + 1] = 3
    ram[BAG_ITEMS_ADDR + 2] = 0x2A
    ram[BAG_ITEMS_ADDR + 3] = 1
    items = decode_bag_items(ram)
    assert len(items) == 2
    assert items[0]["item_id"] == 0x01
    assert items[0]["quantity"] == 3
    assert items[1]["item_hex"] == "0x2A"


def test_decode_party_basic() -> None:
    ram = np.zeros((0x10000,), dtype=np.uint8)
    ram[PARTY_COUNT_ADDR] = 1
    ram[PARTY_LIST_ADDR] = 25
    base = PARTY_MON_BASE
    ram[base] = 25
    ram[base + PARTY_CUR_HP_OFFSET] = 10
    ram[base + PARTY_CUR_HP_OFFSET + 1] = 0
    ram[base + PARTY_MAX_HP_OFFSET] = 20
    ram[base + PARTY_MAX_HP_OFFSET + 1] = 0
    ram[base + PARTY_ACTUAL_LEVEL_OFFSET] = 5
    ram[base + PARTY_STATUS_OFFSET] = 0x04
    party = decode_party(ram)
    assert len(party) == 1
    entry = party[0]
    assert entry["species_id"] == 25
    assert entry["hp"] == 10
    assert entry["max_hp"] == 20
    assert entry["hp_percent"] == 0.5
    assert entry["level"] == 5
    assert entry["status"] == 0x04
