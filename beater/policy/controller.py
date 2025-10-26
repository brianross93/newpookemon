"""Controller with discrete gate over graph ops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from beater.sr_memory import GraphOp


@dataclass(slots=True)
class ControllerConfig:
    latent_dim: int = 256
    hidden_dim: int = 256
    num_skills: int = 4


@dataclass(slots=True)
class ControllerState:
    hidden: torch.Tensor


@dataclass(slots=True)
class ControllerOutput:
    gate_logits: torch.Tensor
    skill_logits: torch.Tensor
    timeout_steps: torch.Tensor
    state: ControllerState


class Controller(nn.Module):
    def __init__(self, config: ControllerConfig):
        super().__init__()
        self.config = config
        self.gru = nn.GRUCell(config.latent_dim, config.hidden_dim)
        self.gate_head = nn.Linear(config.hidden_dim, len(GraphOp))
        self.skill_head = nn.Linear(config.hidden_dim, config.num_skills)
        self.timeout_head = nn.Linear(config.hidden_dim, 1)

    def init_state(self, batch_size: int = 1, device: Optional[torch.device] = None) -> ControllerState:
        hidden = torch.zeros(batch_size, self.config.hidden_dim, device=device)
        return ControllerState(hidden=hidden)

    def forward(self, z: torch.Tensor, state: ControllerState) -> ControllerOutput:
        h = self.gru(z, state.hidden)
        gate_logits = self.gate_head(h)
        skill_logits = self.skill_head(h)
        timeout_steps = torch.relu(self.timeout_head(h)) + 1.0
        next_state = ControllerState(hidden=h)
        return ControllerOutput(
            gate_logits=gate_logits,
            skill_logits=skill_logits,
            timeout_steps=timeout_steps,
            state=next_state,
        )
