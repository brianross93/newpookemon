"""Intrinsic reward helpers for warmup training."""

from __future__ import annotations

import torch

from beater.sr_memory import GraphOp

GRAPH_OPS = list(GraphOp)


def gate_skill_reward(gate_action: torch.Tensor, skill_action: torch.Tensor) -> torch.Tensor:
    """Reward associating with ASSOC/FOLLOW gates to encourage exploration."""
    gate_idx = int(gate_action.item())
    skill_idx = int(skill_action.item())
    gate_op = GRAPH_OPS[gate_idx % len(GRAPH_OPS)]
    reward = 0.1 + 0.05 * skill_idx
    if gate_op in (GraphOp.ASSOC, GraphOp.FOLLOW):
        reward += 0.5
    if gate_op == GraphOp.WRITE:
        reward += 0.2
    return torch.tensor([reward], dtype=torch.float32)
