"""PyBoy-based environment wrapper with savestate ring buffer."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from beater.types import Buttons, Observation, Planlet, Script, ScriptOp

try:
    from pyboy import PyBoy
    from pyboy.utils import WindowEvent
except Exception:  # pragma: no cover - runtime dependency optional
    PyBoy = None  # type: ignore
    WindowEvent = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class EnvConfig:
    rom_path: str
    window: str = "SDL2"
    speed: float = 1.0
    ring_slots: int = 8


class PyBoyEnv:
    """Thin environment facade expected by the rest of the stack."""

    def __init__(self, config: EnvConfig):
        if PyBoy is None:
            raise ImportError("pyboy is required to use PyBoyEnv")

        self.config = config
        self.pyboy = PyBoy(
            config.rom_path,
            window=config.window,
        )
        self.step_idx = 0
        self._ring_slots: List[Optional[bytes]] = [None] * config.ring_slots
        self._ticks_per_advance = max(1, int(round(config.speed)))
        self._ring_cursor = 0
        self._button_events = _build_button_event_map()
        self._save_state()

    # --------------------------------------------------------------------- API
    def observe(self) -> Observation:
        """Grab RGB + RAM snapshot without advancing the emulator."""
        rgb = np.array(self.pyboy.screen.ndarray, copy=True)
        ram = self._snapshot_ram()
        return Observation(
            rgb=rgb, ram=ram, step_idx=self.step_idx, last_save_slot=self._last_slot
        )

    def run_planlet(self, planlet: Planlet) -> Observation:
        """Execute a decoded planlet and return the resulting observation."""
        return self.step_script(planlet.script)

    def step_script(self, script: Script) -> Observation:
        for op in script:
            self._apply_op(op)
            frames = max(op.frames, 0)
            if frames:
                self._advance(frames)
        self._advance(1)  # settle frame after final op
        obs = self.observe()
        self._maybe_save()
        return obs

    def rollback(self, k_back: int = 1) -> None:
        idx = (self._ring_cursor - k_back) % len(self._ring_slots)
        state = self._ring_slots[idx]
        if state is not None:
            with io.BytesIO(state) as f:
                f.seek(0)
                self.pyboy.load_state(f)
            self._ring_cursor = (idx + 1) % len(self._ring_slots)
            LOGGER.info("Rolled back to slot %s", idx)

    def close(self) -> None:
        self.pyboy.stop(save=False)

    # ----------------------------------------------------------------- helpers
    def _advance(self, frames: int) -> None:
        for _ in range(frames):
            for __ in range(self._ticks_per_advance):
                self.pyboy.tick()
                self.step_idx += 1

    def _apply_op(self, op: ScriptOp) -> None:
        if op.op == "WAIT" or op.button in (None, "NOOP"):
            return
        events = self._button_events.get(op.button)
        if not events:
            raise ValueError(f"Unsupported button {op.button}")
        press, release = events
        if op.op == "PRESS":
            self.pyboy.send_input(press)
        elif op.op == "RELEASE":
            self.pyboy.send_input(release)
        else:
            raise ValueError(f"Unknown op {op.op}")

    def _maybe_save(self) -> None:
        # Save every ~512 frames to amortize disk usage.
        if self.step_idx % 512 == 0:
            self._save_state()

    def _save_state(self) -> None:
        with io.BytesIO() as f:
            self.pyboy.save_state(f)
            state = f.getvalue()
        self._ring_slots[self._ring_cursor] = state
        self._last_slot = self._ring_cursor
        self._ring_cursor = (self._ring_cursor + 1) % len(self._ring_slots)

    def _snapshot_ram(self) -> np.ndarray:
        # Access the full 64KB memory space
        return np.array(self.pyboy.memory[0:65536], dtype=np.uint8)


def _build_button_event_map() -> dict[Buttons, tuple[WindowEvent, WindowEvent]]:
    if WindowEvent is None:
        raise ImportError("pyboy is required for button events")
    return {
        "A": (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
        "B": (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B),
        "START": (WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START),
        "SELECT": (
            WindowEvent.PRESS_BUTTON_SELECT,
            WindowEvent.RELEASE_BUTTON_SELECT,
        ),
        "UP": (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP),
        "DOWN": (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
        "LEFT": (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
        "RIGHT": (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
    }
