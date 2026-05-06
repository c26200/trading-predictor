"""Stacked Bidirectional LSTM with Attention for price prediction and direction classification (PyTorch)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AttentionLayer(nn.Module):
    """Custom attention layer for temporal feature weighting."""

    def __init__(self, input_dim: int, attention_dim: int = 32) -> None:
        super().__init__()
        self.W = nn.Linear(input_dim, attention_dim, bias=False)
        self.b = nn.Parameter(torch.zeros(attention_dim))
        self.u = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        v = torch.tanh(self.W(x) + self.b)          # (batch, seq_len, attn_dim)
        vu = self.u(v)                               # (batch, seq_len, 1)
        alphas = F.softmax(vu, dim=1)                # (batch, seq_len, 1)
        output = torch.sum(x * alphas, dim=1)         # (batch, input_dim)
        return output


class LSTMModel(nn.Module):
    """Stacked Bidirectional LSTM with Attention for multi-output prediction."""

    def __init__(
        self,
        n_features: int = 30,
        lstm_units: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.3,
        bidirectional: bool = True,
        attention_dim: int = 32,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.attention_dim = attention_dim

        self.lstm_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        input_size = n_features
        for i, units in enumerate(lstm_units):
            lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=units,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0.0,  # handled manually
            )
            self.lstm_layers.append(lstm)
            lstm_out = units * 2 if bidirectional else units
            if i < len(lstm_units) - 1:
                self.batch_norms.append(nn.BatchNorm1d(lstm_out))
                self.dropouts.append(nn.Dropout(dropout))
            input_size = lstm_out

        # Attention on last LSTM output
        last_units = lstm_units[-1] * 2 if bidirectional else lstm_units[-1]
        self.attention = AttentionLayer(input_dim=last_units, attention_dim=attention_dim)

        # Output heads
        shared_dim = 64
        self.shared_fc1 = nn.Linear(last_units, shared_dim)
        self.shared_drop = nn.Dropout(dropout)
        self.shared_fc2 = nn.Linear(shared_dim, 32)

        self.price_head = nn.Linear(32, 1)
        self.direction_head = nn.Linear(32, 1)

        self._device = torch.device("cpu")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, n_features)
        for i, lstm in enumerate(self.lstm_layers):
            x, _ = lstm(x)
            if i < len(self.lstm_layers) - 1:
                # BatchNorm expects (batch, features, seq_len)
                x = x.permute(0, 2, 1)
                x = self.batch_norms[i](x)
                x = x.permute(0, 2, 1)
                x = self.dropouts[i](x)

        # Attention pooling
        x = self.attention(x)

        # Shared layers
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
        logger.info(f"LSTM model saved to {path}")

    @staticmethod
    def load(path: str) -> Optional[LSTMModel]:
        """Load a saved model from disk. Returns LSTMModel or None."""
        p = Path(path)
        if not p.exists():
            logger.warning(f"Model not found: {path}")
            return None
        try:
            model = torch.load(str(p), map_location="cpu", weights_only=False)
            model.eval()
            logger.info(f"LSTM model loaded from {path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            return None

    def to_device(self, device: torch.device) -> None:
        """Move model to specified device."""
        self._device = device
        self.to(device)
