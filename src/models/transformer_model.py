"""Temporal Transformer with multi-head self-attention for price prediction (PyTorch)."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int = 128, max_len: int = 500) -> None:
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder block with multi-head attention and feed-forward."""

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 8,
        feed_forward_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, feed_forward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class TransformerModel(nn.Module):
    """Temporal Transformer for multi-output price prediction."""

    def __init__(
        self,
        n_features: int = 30,
        d_model: int = 128,
        num_heads: int = 8,
        num_encoder_blocks: int = 4,
        feed_forward_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model=d_model)
        self.input_dropout = nn.Dropout(dropout)

        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                feed_forward_dim=feed_forward_dim,
                dropout=dropout,
            )
            for _ in range(num_encoder_blocks)
        ])

        # Global average pooling over sequence dim
        self.shared_fc1 = nn.Linear(d_model, 64)
        self.shared_drop = nn.Dropout(dropout)
        self.shared_fc2 = nn.Linear(64, 32)

        self.price_head = nn.Linear(32, 1)
        self.direction_head = nn.Linear(32, 1)

        self._device = torch.device("cpu")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, n_features)
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.input_dropout(x)

        for block in self.encoder_blocks:
            x = block(x)

        # Pool over time
        x = x.mean(dim=1)

        x = F.relu(self.shared_fc1(x))
        x = self.shared_drop(x)
        x = F.relu(self.shared_fc2(x))

        price = self.price_head(x)
        direction = torch.sigmoid(self.direction_head(x))
        return price, direction

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (price_predictions, direction_probabilities) as numpy arrays."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32).to(self._device)
            price, direction = self.forward(x)
            return price.cpu().numpy().flatten(), direction.cpu().numpy().flatten()

    def save(self, path: str) -> None:
        """Save full model (architecture + weights) to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, str(p))
        logger.info(f"Transformer model saved to {path}")

    @staticmethod
    def load(path: str) -> Optional[TransformerModel]:
        """Load a saved model from disk. Returns TransformerModel or None."""
        p = Path(path)
        if not p.exists():
            logger.warning(f"Model not found: {path}")
            return None
        try:
            model = torch.load(str(p), map_location="cpu", weights_only=False)
            model.eval()
            logger.info(f"Transformer model loaded from {path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load Transformer model: {e}")
            return None

    def to_device(self, device: torch.device) -> None:
        """Move model to specified device."""
        self._device = device
        self.to(device)
