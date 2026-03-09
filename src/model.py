from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .config import ModelConfig


class SimpleMedPredBackbone(nn.Module):
    """
    A lightweight Transformer-based sequence encoder standing in for
    a full MedPred-style architecture. It maps an RNA sequence to
    per-residue embeddings.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.input_proj = nn.Linear(cfg.vocab_size, cfg.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, 4) one-hot
        lengths: (B,)
        returns: (B, L, d_model)
        """
        h = self.input_proj(x)
        # generate padding mask
        max_len = x.size(1)
        device = x.device
        mask = torch.arange(max_len, device=device)[None, :] >= lengths[:, None]
        h = self.encoder(h, src_key_padding_mask=mask)
        h = self.dropout(h)
        return h


class MedPredHead(nn.Module):
    """
    Simple coordinate regression head:
    per-residue embeddings -> per-residue C1' coords (x,y,z).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, 3)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, L, d_model)
        returns: (B, L, 3)
        """
        return self.fc(h)


class MedPredModel(nn.Module):
    """
    End-to-end baseline model:
    sequence -> per-residue C1' coordinates.

    For compatibility with the competition, we can later extend this
    to output multiple conformations (5 structures) or sampling.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.backbone = SimpleMedPredBackbone(cfg)
        self.head = MedPredHead(cfg.d_model)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, 4)
        lengths: (B,)
        returns: (B, L, 3)
        """
        h = self.backbone(x, lengths)
        coords = self.head(h)
        return coords


def coords_to_flat(coords: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Convert padded (B, L, 3) coords to flattened vectors (B, L*3),
    padding beyond length with zeros.
    """
    b, l, _ = coords.shape
    flat = coords.reshape(b, l * 3)
    # Optionally could mask out positions beyond length; for now rely on loss mask
    return flat

