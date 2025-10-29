"""Structured decoding of Game Boy WRAM for PokAcmon Red/Blue.

This module provides a typed view over the 64 KiB RAM snapshot we receive each
environment step so downstream systems can reason about concrete game states
without sprinkling hard-coded addresses throughout the codebase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Set, Tuple

# --------------------------------------------------------------------------- core

INPUT_STATE_ADDR = 0xD730  # wInputState (joypad handler owner)
TEXT_ENGINE_FLAGS_ADDR = 0xD0A2  # wMainTextControlFlags (bitfield used by engine)
TEXT_STATE_ADDR = 0xC0C4  # wMainTextState (text engine state machine)

# --------------------------------------------------------------------------- naming UI

NAMING_TYPE_ADDR = 0xD07D  # 0=player,1=rival,>=2=nickname, else none
NAMING_SUBSCREEN_ADDR = 0xCC25  # 0x11=grid,0x01=presets, else inactive
NAMING_SUBMIT_ADDR = 0xCC2B  # wNamingScreenSubmitName
NAMING_ALPHABET_CASE_ADDR = 0xD048  # 0 upper,1 lower; other when inactive
NAMING_BUFFER_ADDR = 0xCF4B  # 11-byte buffer
NAMING_BUFFER_MAX = 11

# --------------------------------------------------------------------------- menu UI

MENU_CURSOR_Y_ADDR = 0xCC24
MENU_CURSOR_X_ADDR = 0xCC25
MENU_SELECTED_INDEX_ADDR = 0xCC26
MENU_LAST_ITEM_ID_ADDR = 0xCC28
MENU_PREV_ITEM_ADDR = 0xCC2A
MENU_HIGHLIGHT_SELECT_ADDR = 0xCC35
MENU_FIRST_VISIBLE_ADDR = 0xCC36
MENU_CURSOR_TILE_PTR_ADDR = 0xCC30  # little endian pointer into wTileMap

# --------------------------------------------------------------------------- player/map

PLAYER_Y_ADDR = 0xD361
PLAYER_X_ADDR = 0xD362
MAP_ID_ADDR = 0xD35E
MAP_GROUP_ADDR = 0xD35F

# --------------------------------------------------------------------------- label maps

INPUT_STATE_LABELS: Dict[int, str] = {
    0x00: "disabled",
    0x01: "title_screen",
    0x02: "oak_intro",
    0x03: "overworld",
    0x04: "start_menu",
    0x05: "pokedex",
    0x06: "party_menu",
    0x07: "text_box",
    0x08: "cutscene",
    0x09: "trade_center",
    0x0A: "cinematic",
    0x0B: "naming_screen",
    0x0C: "pc_menu",
    0x0D: "item_storage",
    0x0E: "battle_menu",
    0x0F: "slot_machine",
    0x10: "link_menu",
    0x11: "hall_of_fame",
    0x12: "options_menu",
    0x13: "safari_menu",
    0x14: "pokeflute",
}

TEXT_STATE_LABELS: Dict[int, str] = {
    0x00: "idle",
    0x01: "printing",
    0x02: "await_button",
    0x03: "scroll_wait",
    0x04: "scrolling",
    0x05: "multi_choice",
    0x06: "yes_no_prompt",
    0x07: "auto_close",
    0x08: "finished",
}

# These values are documented in the disassembly; keep a fallback for unknowns.
TEXT_ENGINE_FLAG_MAP: Dict[int, Set[str]] = {
    0x01: {"text_active"},
    0x02: {"text_wait_btn"},
    0x04: {"text_auto_scroll"},
    0x08: {"text_name_input"},
}


LOGGER = logging.getLogger(__name__)
_SEEN_UNKNOWN_INPUT: Set[int] = set()
_SEEN_UNKNOWN_TEXT: Set[int] = set()


def _read_byte(ram: Sequence[int], addr: int) -> int:
    if ram is None or addr >= len(ram):
        return 0
    return int(ram[addr]) & 0xFF


def _read_u16_le(ram: Sequence[int], addr: int) -> int:
    lo = _read_byte(ram, addr)
    hi = _read_byte(ram, addr + 1)
    return lo | (hi << 8)


# --------------------------------------------------------------------------- data classes


@dataclass(slots=True)
class NamingState:
    active: bool
    kind: str
    subscreen: str
    buffer: str
    submit_flag: int
    alphabet_case: int
    raw_kind: int
    raw_subscreen: int


@dataclass(slots=True)
class MenuState:
    open: bool
    cursor: Tuple[int, int]
    selected_index: int
    last_item_id: int
    previous_index: int
    highlighted_index: int
    first_visible: int
    cursor_tile_ptr: int
    item_count: int


@dataclass(slots=True)
class PlayerState:
    x: int
    y: int
    map_group: int
    map_id: int


@dataclass(slots=True)
class RamView:
    input_state_value: int
    input_state_label: str
    text_state_value: int
    text_state_label: str
    text_flags_value: int
    text_flag_labels: Set[str]
    naming: NamingState
    menu: MenuState
    player: PlayerState
    tags: Set[str] = field(default_factory=set)


# --------------------------------------------------------------------------- decoders


def decode_naming_state(ram: Sequence[int]) -> NamingState:
    raw_kind = _read_byte(ram, NAMING_TYPE_ADDR)
    if raw_kind == 0:
        kind = "player"
    elif raw_kind == 1:
        kind = "rival"
    elif raw_kind >= 2:
        kind = "nickname"
    else:
        kind = "none"
    raw_subscreen = _read_byte(ram, NAMING_SUBSCREEN_ADDR)
    if raw_subscreen == 0x11:
        subscreen = "grid"
    elif raw_subscreen == 0x01:
        subscreen = "presets"
    else:
        subscreen = "none"
    submit_flag = _read_byte(ram, NAMING_SUBMIT_ADDR)
    alphabet_case = _read_byte(ram, NAMING_ALPHABET_CASE_ADDR)
    buf_bytes = [
        _read_byte(ram, NAMING_BUFFER_ADDR + i) for i in range(NAMING_BUFFER_MAX)
    ]
    buffer_chars: list[str] = []
    mapping = {
        0x80: "A",
        0x81: "B",
        0x82: "C",
        0x83: "D",
        0x84: "E",
        0x85: "F",
        0x86: "G",
        0x87: "H",
        0x88: "I",
        0x89: "J",
        0x8A: "K",
        0x8B: "L",
        0x8C: "M",
        0x8D: "N",
        0x8E: "O",
        0x8F: "P",
        0x90: "Q",
        0x91: "R",
        0x92: "S",
        0x93: "T",
        0x94: "U",
        0x95: "V",
        0x96: "W",
        0x97: "X",
        0x98: "Y",
        0x99: "Z",
        0x9A: "-",
        0x50: "",
    }
    for value in buf_bytes:
        if value == 0x50:
            break
        buffer_chars.append(mapping.get(value, "?"))
    buffer = "".join(buffer_chars)
    submit_valid = submit_flag in (0, 1)
    alphabet_valid = alphabet_case in (0, 1)
    subscreen_valid = subscreen in ("presets", "grid")
    active = kind != "none" and subscreen_valid and alphabet_valid and submit_valid
    return NamingState(
        active=active,
        kind=kind,
        subscreen=subscreen,
        buffer=buffer,
        submit_flag=submit_flag,
        alphabet_case=alphabet_case,
        raw_kind=raw_kind,
        raw_subscreen=raw_subscreen,
    )


def decode_menu_state(ram: Sequence[int]) -> MenuState:
    cursor = (_read_byte(ram, MENU_CURSOR_Y_ADDR), _read_byte(ram, MENU_CURSOR_X_ADDR))
    selected_index = _read_byte(ram, MENU_SELECTED_INDEX_ADDR)
    last_item_id = _read_byte(ram, MENU_LAST_ITEM_ID_ADDR)
    previous_index = _read_byte(ram, MENU_PREV_ITEM_ADDR)
    highlighted_index = _read_byte(ram, MENU_HIGHLIGHT_SELECT_ADDR)
    first_visible = _read_byte(ram, MENU_FIRST_VISIBLE_ADDR)
    cursor_tile_ptr = _read_u16_le(ram, MENU_CURSOR_TILE_PTR_ADDR)
    # Item count heuristic: last_item_id is 0xFF when no menu, otherwise inclusive max index.
    item_count = 0
    if last_item_id < 0xF0:
        item_count = last_item_id + 1
    open_flag = (
        item_count > 0
        or selected_index < 0xF0
        or cursor_tile_ptr != 0
        or highlighted_index != 0
    )
    return MenuState(
        open=open_flag,
        cursor=cursor,
        selected_index=selected_index,
        last_item_id=last_item_id,
        previous_index=previous_index,
        highlighted_index=highlighted_index,
        first_visible=first_visible,
        cursor_tile_ptr=cursor_tile_ptr,
        item_count=item_count,
    )


def decode_player_state(ram: Sequence[int]) -> PlayerState:
    return PlayerState(
        x=_read_byte(ram, PLAYER_X_ADDR),
        y=_read_byte(ram, PLAYER_Y_ADDR),
        map_group=_read_byte(ram, MAP_GROUP_ADDR),
        map_id=_read_byte(ram, MAP_ID_ADDR),
    )


def decode_ram_view(ram: Sequence[int]) -> RamView:
    input_state_val = _read_byte(ram, INPUT_STATE_ADDR)
    input_state_label = INPUT_STATE_LABELS.get(
        input_state_val, f"unknown_{input_state_val:02X}"
    )
    if input_state_label.startswith("unknown_") and input_state_val not in _SEEN_UNKNOWN_INPUT:
        _SEEN_UNKNOWN_INPUT.add(input_state_val)
        LOGGER.warning("Unknown wInputState value 0x%02X encountered", input_state_val)

    text_state_val = _read_byte(ram, TEXT_STATE_ADDR)
    text_state_label = TEXT_STATE_LABELS.get(
        text_state_val, f"state_{text_state_val:02X}"
    )
    if text_state_label.startswith("state_") and text_state_val not in _SEEN_UNKNOWN_TEXT:
        _SEEN_UNKNOWN_TEXT.add(text_state_val)
        LOGGER.warning("Unknown wMainTextState value 0x%02X encountered", text_state_val)
    text_flags_val = _read_byte(ram, TEXT_ENGINE_FLAGS_ADDR)
    text_flag_labels: Set[str] = set()
    for mask, labels in TEXT_ENGINE_FLAG_MAP.items():
        if text_flags_val & mask:
            text_flag_labels.update(labels)

    naming = decode_naming_state(ram)
    menu = decode_menu_state(ram)
    player = decode_player_state(ram)

    tags: Set[str] = set()
    tags.add(f"input:{input_state_label}")
    if naming.active:
        tags.add("ui:naming")
        tags.add(f"ui:naming:{naming.subscreen}")
    if menu.open:
        tags.add("ui:menu")
    if "text_active" in text_flag_labels or input_state_label in {"text_scroll", "start_menu"}:
        tags.add("ui:textbox")
    if input_state_label == "overworld":
        tags.add("mode:overworld")
    elif input_state_label == "naming_screen":
        tags.add("mode:naming")
    elif input_state_label in {"start_menu", "party_menu", "pokedex"}:
        tags.add("mode:menu")
    elif input_state_label.startswith("unknown_"):
        tags.add("mode:unknown")

    return RamView(
        input_state_value=input_state_val,
        input_state_label=input_state_label,
        text_state_value=text_state_val,
        text_state_label=text_state_label,
        text_flags_value=text_flags_val,
        text_flag_labels=text_flag_labels,
        naming=naming,
        menu=menu,
        player=player,
        tags=tags,
    )
