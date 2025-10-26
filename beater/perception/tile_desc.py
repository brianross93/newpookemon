"""Tile descriptor module for discovery-based passability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class TileDescriptorConfig:
    tile_size: int = 8
    embedding_dim: int = 64
    codebook_size: int = 64


@dataclass(slots=True)
class TileDescriptorOutput:
    embeddings: torch.Tensor  # (B, T, D)
    logits: torch.Tensor  # (B, T, K)
    class_ids: torch.Tensor  # (B, T)
    grid_shape: Tuple[int, int]


class TileDescriptor(nn.Module):
    """Produces tile-level embeddings and unsupervised classes."""

    def __init__(self, config: TileDescriptorConfig):
        super().__init__()
        self.config = config
        embedding_dim = config.embedding_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            nn.Conv2d(128, embedding_dim, kernel_size=1),
        )
        self.pool = nn.AvgPool2d(kernel_size=config.tile_size, stride=config.tile_size)
        self.codebook = nn.Parameter(torch.randn(config.codebook_size, embedding_dim))

    def forward(self, rgb: torch.Tensor) -> TileDescriptorOutput:
        rgb = rgb.float() / 255.0
        feats = self.encoder(rgb)
        tiles = self.pool(feats)  # (B, D, H_t, W_t)
        bsz, dim, h_t, w_t = tiles.shape
        grid_shape = (h_t, w_t)
        tiles = tiles.permute(0, 2, 3, 1).contiguous()
        embeddings = tiles.view(bsz, -1, dim)
        codebook_norm = F.normalize(self.codebook, dim=-1)
        emb_norm = F.normalize(embeddings, dim=-1)
        logits = emb_norm @ codebook_norm.t()
        class_ids = torch.argmax(logits, dim=-1)
        return TileDescriptorOutput(
            embeddings=embeddings,
            logits=logits,
            class_ids=class_ids,
            grid_shape=grid_shape,
        )

    @staticmethod
    def class_tokens(class_ids: torch.Tensor, grid_shape: Tuple[int, int]) -> List[List[str]]:
        """Convert class ids into human-readable tokens per sample."""
        bsz, total = class_ids.shape
        h_t, w_t = grid_shape
        assert total == h_t * w_t
        tokens: List[List[str]] = []
        for sample in class_ids.cpu().tolist():
            tokens.append([f"class_{idx}" for idx in sample])
        return tokens

    def tile_keys(
        self, class_ids: torch.Tensor, grid_shape: Tuple[int, int], room_id: str = "global"
    ) -> List[List[str]]:
        """Return stable keys that blend room + tile class for passability."""
        tokens = self.class_tokens(class_ids, grid_shape)
        return [[f"{room_id}:{tok}" for tok in sample] for sample in tokens]

    @staticmethod
    def reshape_tokens(tokens: Sequence[str], grid_shape: Tuple[int, int]) -> List[List[str]]:
        """Reshape a flat sequence of tokens into a (H, W) grid."""
        h_t, w_t = grid_shape
        if len(tokens) != h_t * w_t:
            raise ValueError("Token count does not match grid size")
        grid: List[List[str]] = []
        for row in range(h_t):
            start = row * w_t
            grid.append(list(tokens[start : start + w_t]))
        return grid
