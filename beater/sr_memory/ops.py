"""Graph operation primitives for the controller gate."""

from __future__ import annotations

from enum import Enum, auto


class GraphOp(Enum):
    ENCODE = auto()
    ASSOC = auto()
    FOLLOW = auto()
    WRITE = auto()
    HALT = auto()


__all__ = ["GraphOp"]
