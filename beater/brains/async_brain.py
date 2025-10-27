"""Async wrapper for GPTBrain to avoid blocking the control loop.

Schedules suggest_goal calls on a background thread and lets callers poll
for results at safe boundaries (e.g., after a planlet completes).
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .gpt import GPTBrain, GoalSuggestion

Coord = Tuple[int, int]


@dataclass(slots=True)
class PendingJob:
    future: Future
    step_idx: int
    candidates_snapshot: Sequence[Coord]


class AsyncBrain:
    def __init__(self, brain: GPTBrain, max_workers: int = 1):
        self._brain = brain
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="brain")
        self._lock = threading.Lock()
        self._pending: Optional[PendingJob] = None

    def has_pending(self) -> bool:
        with self._lock:
            return self._pending is not None and not self._pending.future.done()

    def request(
        self,
        tile_grid: List[List[str]],
        candidates: List[Coord],
        context: Optional[Dict[str, object]],
        step_idx: int,
        image_bytes: Optional[bytes] = None,
    ) -> bool:
        """Schedule a background suggest_goal call. Returns False if one is pending."""
        with self._lock:
            if self._pending is not None and not self._pending.future.done():
                return False
            fut = self._pool.submit(self._brain.suggest_goal, tile_grid, candidates, context, image_bytes)
            self._pending = PendingJob(future=fut, step_idx=step_idx, candidates_snapshot=list(candidates))
            return True

    def poll(self) -> Optional[Tuple[GoalSuggestion, int, Sequence[Coord]]]:
        """Return (directive, step_idx, candidates_snapshot) when ready; otherwise None."""
        with self._lock:
            job = self._pending
            if job is None:
                return None
            if not job.future.done():
                return None
            self._pending = None
        try:
            result = job.future.result()
        except Exception:
            return None
        if result is None:
            return None
        return result, job.step_idx, job.candidates_snapshot

    def cancel(self) -> None:
        with self._lock:
            if self._pending and not self._pending.future.done():
                self._pending.future.cancel()
            self._pending = None

    def shutdown(self) -> None:
        with self._lock:
            self._pending = None
        self._pool.shutdown(wait=False, cancel_futures=True)
