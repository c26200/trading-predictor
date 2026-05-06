"""Training pipeline with PyTorch: early stopping, LR scheduling, checkpointing, GPU auto-detect."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping that tracks a metric and restores best weights."""

    def __init__(self, patience: int = 15, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_state: Optional[dict] = None
        self.stopped = False

    def step(self, score: float, model: nn.Module) -> bool:
        """Return True if should stop."""
        if self.best_score is None or score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        if self.counter >= self.patience:
            self.stopped = True
            return True
        return False

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class ModelTrainer:
    """Handles model training with configurable callbacks and checkpointing."""

    def __init__(
        self,
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.15,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 15,
        reduce_lr_patience: int = 7,
        reduce_lr_factor: float = 0.5,
        min_lr: float = 0.00001,
        models_dir: str = "models/saved",
        history_dir: str = "models/saved/history",
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_factor = reduce_lr_factor
        self.min_lr = min_lr
        self.models_dir = Path(models_dir)
        self.history_dir = Path(history_dir)
        self._device = self._detect_device()

    def _detect_device(self) -> torch.device:
        """Detect available hardware (GPU/CPU)."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Apple MPS detected")
        else:
            device = torch.device("cpu")
            logger.info("No GPU found. Using CPU.")
        return device

    def train_model(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        ticker: str = "UNKNOWN",
    ) -> dict:
        """Train a PyTorch model and return history dict.

        Args:
            model: PyTorch nn.Module (LSTMModel or TransformerModel).
            X_train: Training sequences (n_samples, seq_len, n_features).
            y_train: Target array [price, direction] (n_samples, 2).
            ticker: Ticker symbol for logging and checkpoint naming.
        """
        model.to_device(self._device)
        model.train()

        # Split data
        n_val = max(1, int(len(X_train) * self.validation_split))
        indices = np.random.permutation(len(X_train))
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        X_tr = torch.tensor(X_train[train_idx], dtype=torch.float32)
        y_tr = torch.tensor(y_train[train_idx], dtype=torch.float32)
        X_val = torch.tensor(X_train[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y_train[val_idx], dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=self.batch_size * 2)

        optimizer = Adam(model.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.reduce_lr_factor,
            patience=self.reduce_lr_patience,
            min_lr=self.min_lr,
        )
        price_criterion = nn.MSELoss()
        dir_criterion = nn.BCELoss()
        early_stop = EarlyStopping(patience=self.early_stopping_patience)

        history: dict[str, list[float]] = {
            "loss": [], "val_loss": [], "price_loss": [], "dir_loss": [],
            "val_price_loss": [], "val_dir_loss": [], "lr": [],
        }

        logger.info(f"Training {ticker} on {self._device} | Epochs: {self.epochs} | Samples: {len(train_idx)}")

        for epoch in range(self.epochs):
            # ── Training ──
            model.train()
            epoch_loss, epoch_p_loss, epoch_d_loss = 0.0, 0.0, 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self._device)
                batch_y = batch_y.to(self._device)

                optimizer.zero_grad()
                price_pred, dir_pred = model(batch_X)

                p_loss = price_criterion(price_pred.squeeze(-1), batch_y[:, 0])
                d_loss = dir_criterion(dir_pred.squeeze(-1), batch_y[:, 1])
                loss = p_loss + 0.5 * d_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(batch_X)
                epoch_p_loss += p_loss.item() * len(batch_X)
                epoch_d_loss += d_loss.item() * len(batch_X)

            n_train = len(train_idx)
            history["loss"].append(epoch_loss / n_train)
            history["price_loss"].append(epoch_p_loss / n_train)
            history["dir_loss"].append(epoch_d_loss / n_train)

            # ── Validation ──
            model.eval()
            val_loss, val_p_loss, val_d_loss = 0.0, 0.0, 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self._device)
                    batch_y = batch_y.to(self._device)

                    price_pred, dir_pred = model(batch_X)
                    p_loss = price_criterion(price_pred.squeeze(-1), batch_y[:, 0])
                    d_loss = dir_criterion(dir_pred.squeeze(-1), batch_y[:, 1])
                    loss = p_loss + 0.5 * d_loss

                    val_loss += loss.item() * len(batch_X)
                    val_p_loss += p_loss.item() * len(batch_X)
                    val_d_loss += d_loss.item() * len(batch_X)

            n_val_actual = len(val_idx)
            val_loss_avg = val_loss / n_val_actual
            history["val_loss"].append(val_loss_avg)
            history["val_price_loss"].append(val_p_loss / n_val_actual)
            history["val_dir_loss"].append(val_d_loss / n_val_actual)
            history["lr"].append(optimizer.param_groups[0]["lr"])

            # ── Scheduler & early stopping ──
            scheduler.step(val_loss_avg)

            if early_stop.step(val_loss_avg, model):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % max(1, self.epochs // 5) == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} — "
                    f"Loss: {history['loss'][-1]:.4f} | "
                    f"Val Loss: {val_loss_avg:.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )

        # Restore best weights
        if early_stop.best_state is not None:
            early_stop.restore(model)
            logger.info("Restored best model weights")

        # Save history
        history_out = {k: [float(v) for v in vals] for k, vals in history.items()}
        history_out["ticker"] = ticker
        history_out["device"] = str(self._device)
        history_out["timestamp"] = datetime.now().isoformat()
        history_out["epochs_completed"] = len(history["loss"])

        self.history_dir.mkdir(parents=True, exist_ok=True)
        hist_path = self.history_dir / f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(hist_path, "w") as f:
            json.dump(history_out, f, indent=2)
        logger.info(f"Training history saved to {hist_path}")

        return history_out

    def train_all(
        self,
        X_dict: dict[str, np.ndarray],
        y_dict: dict[str, np.ndarray],
        model_builder: Callable,
    ) -> dict[str, dict]:
        """Train models for all tickers sequentially.

        Args:
            X_dict: Dict mapping ticker -> X_train array.
            y_dict: Dict mapping ticker -> y_train array.
            model_builder: Callable that returns (lstm_model, transformer_model) tuple.

        Returns:
            Dict mapping ticker -> training history dict.
        """
        results: dict[str, dict] = {}
        for ticker in X_dict:
            if ticker not in y_dict:
                logger.warning(f"No targets for {ticker}, skipping.")
                continue
            logger.info(f"=== Training models for {ticker} ===")
            try:
                lstm, transformer = model_builder()

                logger.info(f"Training LSTM for {ticker}...")
                lstm_hist = self.train_model(lstm, X_dict[ticker], y_dict[ticker], f"{ticker}_lstm")

                logger.info(f"Training Transformer for {ticker}...")
                trans_hist = self.train_model(transformer, X_dict[ticker], y_dict[ticker], f"{ticker}_transformer")

                lstm.save(str(self.models_dir / ticker / "lstm_model.pt"))
                transformer.save(str(self.models_dir / ticker / "transformer_model.pt"))

                results[ticker] = {
                    "lstm_history": lstm_hist,
                    "transformer_history": trans_hist,
                }
            except Exception as e:
                logger.error(f"Failed to train models for {ticker}: {e}")
                results[ticker] = {"error": str(e)}

        return results

    @property
    def device(self) -> str:
        return str(self._device)
