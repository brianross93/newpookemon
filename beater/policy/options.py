"""Lightweight option mining utilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Counter as CounterType, Optional

from beater.types import PlanletKind


@dataclass(slots=True)
class OptionBank:
    min_count: int = 5
    _counter: CounterType[PlanletKind] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_counter", Counter())

    def record(self, kind: PlanletKind) -> None:
        self._counter[kind] += 1

    def suggest(self) -> Optional[PlanletKind]:
        total = sum(self._counter.values())
        if total < self.min_count or not self._counter:
            return None
        return max(self._counter, key=self._counter.get)

    def bias_vector(self, vocab: list[PlanletKind]) -> list[float]:
        suggestion = self.suggest()
        if suggestion is None:
            return [0.0] * len(vocab)
        return [0.5 if kind == suggestion else 0.0 for kind in vocab]
