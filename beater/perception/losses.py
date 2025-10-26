"""Perception auxiliary losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def tile_consistency_loss(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """Symmetric KL encouraging consistent tile classes across augmentations."""
    log_pa = torch.log_softmax(logits_a, dim=-1)
    log_pb = torch.log_softmax(logits_b, dim=-1)
    pa = log_pa.exp()
    pb = log_pb.exp()
    loss_ab = F.kl_div(log_pa, pb, reduction="batchmean", log_target=False)
    loss_ba = F.kl_div(log_pb, pa, reduction="batchmean", log_target=False)
    return 0.5 * (loss_ab + loss_ba)


def tile_entropy_regularizer(logits: torch.Tensor, target: float = 0.7) -> torch.Tensor:
    """Encourage moderately high-entropy tile assignments."""
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * probs.clamp_min(1e-6).log()).sum(dim=-1).mean()
    return (entropy - target).abs()


def prototype_separation_loss(codebook: torch.Tensor) -> torch.Tensor:
    """Penalize prototype collapse by pushing cosine sims toward identity."""
    norm = F.normalize(codebook, dim=-1)
    sim = norm @ norm.t()
    eye = torch.eye(sim.size(0), device=sim.device)
    return (sim - eye).abs().mean()
