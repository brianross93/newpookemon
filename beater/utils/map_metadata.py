"""ROM-backed map metadata loader for portals and interactables."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

__all__ = [
    "MapMetadataLoader",
    "MapPortal",
    "MapInteractable",
]


def _parse_int(token: str) -> Optional[int]:
    token = token.strip()
    if not token:
        return None
    if token.startswith("$"):
        try:
            return int(token[1:], 16)
        except ValueError:
            return None
    try:
        return int(token, 0)
    except ValueError:
        return None


def _clean_label(*parts: str) -> str:
    return "_".join(p for p in parts if p).lower()


@dataclass(slots=True)
class MapPortal:
    x: int
    y: int
    dest_map: str
    dest_warp: int
    label: str


@dataclass(slots=True)
class MapInteractable:
    x: int
    y: int
    label: str
    kind: str
    facing: Optional[str] = None
    routine: Optional[str] = None
    text_id: Optional[str] = None


class MapMetadataLoader:
    """Lazy parser for Gen-1 map headers (warp tiles, hidden objects, etc.)."""

    def __init__(self, pokered_root: Path):
        self._root = pokered_root
        self._map_names: Dict[int, str] = {}
        self._map_const: Dict[str, str] = {}
        self._hidden_objects: Dict[str, List[MapInteractable]] = {}
        self._tileset_collisions: Dict[str, Set[int]] = self._load_tileset_collisions()
        self._map_tileset_cache: Dict[int, Optional[str]] = {}

    # ------------------------------------------------------------------ public
    def get_portals(self, map_id: int) -> List[MapPortal]:
        name = self._map_names_lookup().get(map_id)
        if not name:
            return []
        return list(self._build_map_metadata(name)["portals"])

    def get_interactables(self, map_id: int) -> List[MapInteractable]:
        name = self._map_names_lookup().get(map_id)
        if not name:
            return []
        return list(self._build_map_metadata(name)["interactables"])

    def map_label(self, map_id: int) -> Optional[str]:
        name = self._map_names_lookup().get(map_id)
        if not name:
            return None
        const = self._map_const_lookup().get(name)
        return const or name

    def tileset_for_map(self, map_id: int) -> Optional[str]:
        if map_id in self._map_tileset_cache:
            return self._map_tileset_cache[map_id]
        name = self._map_names_lookup().get(map_id)
        if not name:
            self._map_tileset_cache[map_id] = None
            return None
        tileset = self._parse_map_header_tileset(name)
        self._map_tileset_cache[map_id] = tileset
        return tileset

    def collision_ids(self, map_id: int) -> Set[int]:
        tileset = self.tileset_for_map(map_id)
        if not tileset:
            return set()
        key = self._normalize_tileset_label(tileset)
        return set(self._tileset_collisions.get(key, set()))

    # ---------------------------------------------------------------- internal
    @lru_cache(maxsize=None)
    def _build_map_metadata(self, map_name: str) -> Dict[str, Iterable]:
        portals: List[MapPortal] = []
        interactables: List[MapInteractable] = []

        # Warps / object definitions
        objects_path = self._root / "data" / "maps" / "objects" / f"{map_name}.asm"
        if objects_path.exists():
            portals.extend(self._parse_warps(objects_path, map_name))
            interactables.extend(self._parse_bg_events(objects_path, map_name))
            interactables.extend(self._parse_object_events(objects_path, map_name))

        # Hidden objects (PCs, SNES, etc.)
        interactables.extend(self._hidden_objects_lookup().get(map_name, []))

        return {"portals": portals, "interactables": interactables}

    def _map_names_lookup(self) -> Dict[int, str]:
        if self._map_names:
            return self._map_names
        pointers = self._root / "data" / "maps" / "map_header_pointers.asm"
        if not pointers.exists():
            return self._map_names
        pattern = re.compile(r"dw\s+([A-Za-z0-9_]+)_h")
        idx = 0
        for line in pointers.read_text(encoding="ascii", errors="ignore").splitlines():
            match = pattern.search(line)
            if match:
                self._map_names[idx] = match.group(1)
                idx += 1
        return self._map_names

    def _map_const_lookup(self) -> Dict[str, str]:
        if self._map_const:
            return self._map_const
        headers_dir = self._root / "data" / "maps" / "headers"
        pattern = re.compile(r"map_header\s+([A-Za-z0-9_]+),\s*([A-Z0-9_]+)")
        if not headers_dir.exists():
            return self._map_const
        for path in headers_dir.glob("*.asm"):
            for line in path.read_text(encoding="ascii", errors="ignore").splitlines():
                match = pattern.search(line)
                if match:
                    self._map_const[match.group(1)] = match.group(2)
                    break
        return self._map_const

    def _hidden_objects_lookup(self) -> Dict[str, List[MapInteractable]]:
        if self._hidden_objects:
            return self._hidden_objects
        hidden_path = self._root / "data" / "events" / "hidden_objects.asm"
        if not hidden_path.exists():
            return self._hidden_objects
        current: Optional[str] = None
        section: List[MapInteractable] = []
        section_re = re.compile(r"^([A-Za-z0-9_]+)HiddenObjects:")
        object_re = re.compile(
            r"hidden_object\s+([$\w]+),\s*([$\w]+),\s*([A-Z0-9_]+),\s*([A-Za-z0-9_]+)"
        )
        for raw_line in hidden_path.read_text(encoding="ascii", errors="ignore").splitlines():
            line = raw_line.split(";", 1)[0].strip()
            if not line:
                continue
            section_match = section_re.match(line)
            if section_match:
                if current and section:
                    self._hidden_objects[current] = list(section)
                current = section_match.group(1)
                section = []
                continue
            if line.startswith("db -1"):
                if current and section:
                    self._hidden_objects[current] = list(section)
                current = None
                section = []
                continue
            obj_match = object_re.match(line)
            if obj_match and current:
                x_tok, y_tok, facing, routine = obj_match.groups()
                x_val = _parse_int(x_tok)
                y_val = _parse_int(y_tok)
                if x_val is None or y_val is None:
                    continue
                label = _clean_label(current, routine)
                section.append(
                    MapInteractable(
                        x=x_val,
                        y=y_val,
                        label=label,
                        kind="hidden_object",
                        facing=facing,
                        routine=routine,
                    )
                )
        if current and section:
            self._hidden_objects[current] = list(section)
        return self._hidden_objects

    # ---------------------------- per-map parsers -----------------------------
    def _parse_warps(self, path: Path, map_name: str) -> List[MapPortal]:
        portals: List[MapPortal] = []
        pattern = re.compile(
            r"warp_event\s+([$\w]+),\s*([$\w]+),\s*([A-Z0-9_+-]+),\s*([-\w]+)"
        )
        for raw_line in path.read_text(encoding="ascii", errors="ignore").splitlines():
            line = raw_line.split(";", 1)[0].strip()
            if not line:
                continue
            match = pattern.match(line)
            if not match:
                continue
            x_tok, y_tok, dest_map, dest_warp_tok = match.groups()
            x_val = _parse_int(x_tok)
            y_val = _parse_int(y_tok)
            dest_warp = _parse_int(dest_warp_tok)
            if x_val is None or y_val is None or dest_warp is None:
                continue
            label = _clean_label(map_name, "warp", str(len(portals)))
            portals.append(
                MapPortal(
                    x=x_val,
                    y=y_val,
                    dest_map=dest_map,
                    dest_warp=dest_warp,
                    label=label,
                )
            )
        return portals

    def _parse_bg_events(self, path: Path, map_name: str) -> List[MapInteractable]:
        interactables: List[MapInteractable] = []
        pattern = re.compile(r"bg_event\s+([$\w]+),\s*([$\w]+),\s*([A-Z0-9_]+)")
        count = 0
        for raw_line in path.read_text(encoding="ascii", errors="ignore").splitlines():
            line = raw_line.split(";", 1)[0].strip()
            if not line:
                continue
            match = pattern.match(line)
            if not match:
                continue
            x_tok, y_tok, text_id = match.groups()
            x_val = _parse_int(x_tok)
            y_val = _parse_int(y_tok)
            if x_val is None or y_val is None:
                continue
            label = _clean_label(map_name, "bg", str(count))
            count += 1
            interactables.append(
                MapInteractable(
                    x=x_val,
                    y=y_val,
                    label=label,
                    kind="bg_event",
                    text_id=text_id,
                )
            )
        return interactables

    def _parse_object_events(self, path: Path, map_name: str) -> List[MapInteractable]:
        interactables: List[MapInteractable] = []
        pattern = re.compile(
            r"object_event\s+([$\w]+),\s*([$\w]+),\s*([A-Z0-9_]+),\s*([A-Z0-9_]+),\s*([A-Z0-9_]+),\s*([A-Z0-9_]+)"
        )
        count = 0
        for raw_line in path.read_text(encoding="ascii", errors="ignore").splitlines():
            line = raw_line.split(";", 1)[0].strip()
            if not line:
                continue
            match = pattern.match(line)
            if not match:
                continue
            x_tok, y_tok, sprite, movement, direction, text_id = match.groups()
            x_val = _parse_int(x_tok)
            y_val = _parse_int(y_tok)
            if x_val is None or y_val is None:
                continue
            label = _clean_label(map_name, "obj", str(count), text_id)
            count += 1
            interactables.append(
                MapInteractable(
                    x=x_val,
                    y=y_val,
                    label=label,
                    kind="object_event",
                    facing=direction,
                    text_id=text_id,
                    routine=sprite,
                )
                )
        return interactables

    # ----------------------------- tileset helpers ----------------------------
    def _load_tileset_collisions(self) -> Dict[str, Set[int]]:
        collisions: Dict[str, Set[int]] = {}
        path = self._root / "data" / "tilesets" / "collision_tile_ids.asm"
        if not path.exists():
            return collisions
        pending_labels: List[str] = []
        for raw_line in path.read_text(encoding="ascii", errors="ignore").splitlines():
            line = raw_line.split(";", 1)[0].strip()
            if not line:
                continue
            if line.endswith("::"):
                pending_labels.append(line[:-2])
                continue
            if not line.startswith("coll_tiles"):
                continue
            args = line[len("coll_tiles") :].strip()
            values: List[int] = []
            if args:
                tokens = [tok.strip() for tok in args.split(",") if tok.strip()]
                for token in tokens:
                    if token.startswith("$"):
                        values.append(int(token[1:], 16))
                    else:
                        values.append(int(token, 0))
            for label in pending_labels or ["_UNNAMED_"]:
                key = self._normalize_tileset_label(label)
                collisions[key] = set(values)
            pending_labels.clear()
        return collisions

    def _normalize_tileset_label(self, label: str) -> str:
        name = re.sub(r"_Coll$", "", label, flags=re.IGNORECASE)
        # Convert CamelCase (and digits) to uppercase underscore format.
        parts: List[str] = []
        token = ""
        for idx, ch in enumerate(name):
            if idx > 0 and (
                (ch.isupper() and (name[idx - 1].islower() or (idx + 1 < len(name) and name[idx + 1].islower())))
                or (ch.isdigit() and not name[idx - 1].isdigit())
            ):
                parts.append(token)
                token = ch
            else:
                token += ch
        if token:
            parts.append(token)
        return "_".join(part.upper() for part in parts if part)

    def _parse_map_header_tileset(self, map_name: str) -> Optional[str]:
        header_path = self._root / "data" / "maps" / "headers" / f"{map_name}.asm"
        if not header_path.exists():
            return None
        for raw_line in header_path.read_text(encoding="ascii", errors="ignore").splitlines():
            line = raw_line.split(";", 1)[0].strip()
            if not line or not line.startswith("map_header"):
                continue
            parts = [part.strip() for part in line[len("map_header") :].split(",")]
            if len(parts) >= 3:
                return parts[2]
        return None
