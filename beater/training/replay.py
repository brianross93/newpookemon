"""Simple rollout buffer for PPO updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass(slots=True)
class RolloutStep:
    latent: torch.Tensor
    hidden: torch.Tensor
    gate_action: torch.Tensor
    skill_action: torch.Tensor
    old_log_prob: torch.Tensor
    reward: torch.Tensor


@dataclass(slots=True)
class RolloutBatch:
    latents: torch.Tensor
    hidden: torch.Tensor
    gate_actions: torch.Tensor
    skill_actions: torch.Tensor
    old_log_probs: torch.Tensor
    rewards: torch.Tensor


class RolloutBuffer:
    def __init__(self) -> None:
        self.steps: List[RolloutStep] = []

    def add(self, step: RolloutStep) -> None:
        self.steps.append(step)

    def __len__(self) -> int:
        return len(self.steps)

    def to_batch(self) -> RolloutBatch:
        if not self.steps:
            raise ValueError("Cannot convert empty buffer")
        latents = torch.stack([s.latent for s in self.steps])
        hidden = torch.stack([s.hidden for s in self.steps])
        gate_actions = torch.stack([s.gate_action for s in self.steps])
        skill_actions = torch.stack([s.skill_action for s in self.steps])
        old_log_probs = torch.stack([s.old_log_prob for s in self.steps])
        rewards = torch.stack([s.reward for s in self.steps])
        return RolloutBatch(
            latents=latents,
            hidden=hidden,
            gate_actions=gate_actions,
            skill_actions=skill_actions,
            old_log_probs=old_log_probs,
            rewards=rewards,
        )
