"""Slot-friendly perception stack with temporal smoothing and RND aux."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class PerceptionConfig:
    z_dim: int = 256
    smooth_tau: float = 0.9
    rnd_dim: int = 128
    use_ram: bool = True


class FrameEncoder(nn.Module):
    """Simple CNN encoder mixing RGB frames and RAM bytes."""

    def __init__(self, z_dim: int, use_ram: bool = True) -> None:
        super().__init__()
        in_ch = 4  # PyBoy returns RGBA.
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
        )
        conv_out = 128 * 18 * 20  # Down-sampled 144x160 by /8 (approx).
        ram_dim = 256 if use_ram else 0
        self.use_ram = use_ram
        self.ram_proj = (
            nn.Sequential(
                nn.Linear(65536, 1024),
                nn.SiLU(),
                nn.Linear(1024, ram_dim),
                nn.LayerNorm(ram_dim),
            )
            if use_ram
            else None
        )
        self.fc = nn.Sequential(
            nn.Linear(conv_out + ram_dim, 512),
            nn.SiLU(),
            nn.Linear(512, z_dim),
        )

    def forward(self, rgb: torch.Tensor, ram: Optional[torch.Tensor]) -> torch.Tensor:
        rgb = rgb.float() / 255.0
        feats = self.conv(rgb)
        feats = feats.flatten(start_dim=1)
        if self.use_ram and ram is not None:
            ram = ram.float() / 255.0
            ram_flat = ram.view(ram.shape[0], -1)
            ram_feat = self.ram_proj(ram_flat)
            feats = torch.cat([feats, ram_feat], dim=-1)
        return self.fc(feats)


class TemporalSmoother:
    """EMA smoother over latent features to stabilize policy inputs."""

    def __init__(self, tau: float = 0.9):
        self.tau = tau
        self._state: Dict[str, torch.Tensor] = {}

    def reset(self, context: Optional[str] = None) -> None:
        if context is None:
            self._state.clear()
        else:
            self._state.pop(context, None)

    def __call__(self, z: torch.Tensor, context: Optional[str] = None) -> torch.Tensor:
        key = context or "__default__"
        if key not in self._state:
            self._state[key] = z.detach()
        else:
            self._state[key] = self.tau * self._state[key] + (1.0 - self.tau) * z.detach()
        return self._state[key]


class _RNDHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RandomNetworkDistillation(nn.Module):
    """RND auxiliary for intrinsic motivation + representation shaping."""

    def __init__(self, z_dim: int, rnd_dim: int):
        super().__init__()
        self.target = _RNDHead(z_dim, rnd_dim)
        self.predictor = _RNDHead(z_dim, rnd_dim)
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            target = self.target(z)
        pred = self.predictor(z)
        loss = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
        return loss, pred


class Perception(nn.Module):
    """High-level perception facade matching the PROJECTOVERVIEW contract."""

    def __init__(self, config: PerceptionConfig):
        super().__init__()
        self.config = config
        self.encoder = FrameEncoder(config.z_dim, use_ram=config.use_ram)
        self.rnd = RandomNetworkDistillation(config.z_dim, config.rnd_dim)
        self.smoother = TemporalSmoother(config.smooth_tau)

    def reset(self, context: Optional[str] = None) -> None:
        self.smoother.reset(context)

    def forward(
        self, rgb: torch.Tensor, ram: Optional[torch.Tensor], *, context: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(rgb, ram)
        z_smooth = self.smoother(z, context=context)
        rnd_loss, _ = self.rnd(z)
        return z_smooth, rnd_loss.detach()
