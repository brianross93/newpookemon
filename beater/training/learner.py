"""Minimal PPO trainer for controller warmup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from beater.policy import AffordancePrior, Controller, ControllerState
from beater.training.replay import RolloutBuffer


@dataclass(slots=True)
class PPOConfig:
    lr: float = 3e-4
    clip_eps: float = 0.1
    entropy_coef: float = 0.01


class PPOTrainer:
    def __init__(self, controller: Controller, affordance: AffordancePrior, config: PPOConfig):
        self.controller = controller
        self.affordance = affordance
        self.config = config
        self.optimizer = torch.optim.Adam(controller.parameters(), lr=config.lr)

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        if len(buffer) == 0:
            return {"policy_loss": 0.0, "entropy": 0.0}
        batch = buffer.to_batch()
        rewards = batch.rewards
        advantages = rewards - rewards.mean()
        advantages = advantages / (advantages.std() + 1e-5)
        policy_losses = []
        entropies = []
        for idx in range(len(buffer)):
            latent = batch.latents[idx].unsqueeze(0)
            hidden = batch.hidden[idx].unsqueeze(0)
            gate_action = batch.gate_actions[idx]
            skill_action = batch.skill_actions[idx]
            old_log_prob = batch.old_log_probs[idx]
            ctrl_out = self.controller(latent, ControllerState(hidden=hidden))
            affordance_logits = self.affordance(latent)
            gate_log_probs = F.log_softmax(ctrl_out.gate_logits, dim=-1)
            skill_log_probs = F.log_softmax(ctrl_out.skill_logits + affordance_logits, dim=-1)
            gate_lp = gate_log_probs[0, gate_action]
            skill_lp = skill_log_probs[0, skill_action]
            log_prob = gate_lp + skill_lp
            ratio = torch.exp(log_prob - old_log_prob)
            clipped = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps)
            advantage = advantages[idx]
            policy_loss = -torch.min(ratio * advantage, clipped * advantage)
            policy_losses.append(policy_loss)
            entropy = -(gate_log_probs.exp() * gate_log_probs).sum(-1)
            entropy += -(skill_log_probs.exp() * skill_log_probs).sum(-1)
            entropies.append(entropy.mean())

        loss = torch.stack(policy_losses).mean()
        entropy_term = torch.stack(entropies).mean()
        total_loss = loss - self.config.entropy_coef * entropy_term
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return {
            "policy_loss": float(loss.item()),
            "entropy": float(entropy_term.item()),
        }
