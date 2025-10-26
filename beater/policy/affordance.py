"""Affordance prior providing skill biases."""

from __future__ import annotations

import torch
import torch.nn as nn


class AffordancePrior(nn.Module):
    def __init__(self, latent_dim: int, num_skills: int):
        super().__init__()
        hidden = max(64, latent_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_skills),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
