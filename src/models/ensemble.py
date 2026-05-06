"""Weighted ensemble combining LSTM + Transformer with dynamic confidence scoring (PyTorch)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Ensemble of LSTM and Transformer models with dynamic weighting."""

    def __init__(
        self,
        lstm_model=None,  # LSTMModel
        transformer_model=None,  # TransformerModel
        lstm_weight: float = 0.5,
        transformer_weight: float = 0.5,
        confidence_threshold: float = 75.0,
        models_dir: str = "models/saved",
    ) -> None:
        self.lstm = lstm_model
        self.transformer = transformer_model
        self.lstm_weight = lstm_weight
        self.transformer_weight = transformer_weight
        self.confidence_threshold = confidence_threshold
        self.models_dir = Path(models_dir)

        # Performance tracking for dynamic weighting
        self._lstm_errors: list[float] = []
        self._transformer_errors: list[float] = []
        self._recent_predictions: list[dict] = []

    def predict(self, X: np.ndarray) -> dict:
        """Run ensemble prediction and return structured result.

        Returns dict with keys:
            predicted_price, direction, confidence, lstm_price, transformer_price,
            lstm_dir_prob, transformer_dir_prob, signal_strength
        """
        if self.lstm is None or self.transformer is None:
            raise RuntimeError("Models not set. Load or assign lstm/transformer first.")

        lstm_price, lstm_dir = self.lstm.predict(X)
        trans_price, trans_dir = self.transformer.predict(X)

        # Weighted ensemble
        w_l = self.lstm_weight
        w_t = self.transformer_weight
        total_w = w_l + w_t
        if total_w == 0:
            total_w = 1.0

        ensemble_price = (w_l * lstm_price[-1] + w_t * trans_price[-1]) / total_w
        ensemble_dir_prob = (w_l * lstm_dir[-1] + w_t * trans_dir[-1]) / total_w

        # Confidence scoring
        price_agreement = 1.0 - abs(lstm_price[-1] - trans_price[-1]) / (abs(ensemble_price) + 0.01)
        dir_agreement = 1.0 - abs(lstm_dir[-1] - trans_dir[-1])
        model_confidence = min(price_agreement, dir_agreement)
        confidence = max(0.0, min(100.0, model_confidence * 100.0))

        # Direction classification
        if ensemble_dir_prob > 0.55:
            direction = "UP"
        elif ensemble_dir_prob < 0.45:
            direction = "DOWN"
        else:
            direction = "NEUTRAL"

        # Signal strength (1-5 stars)
        star_raw = confidence / 20.0 + abs(ensemble_dir_prob - 0.5) * 10
        signal_strength = max(1, min(5, int(round(star_raw))))

        result = {
            "predicted_price": float(ensemble_price),
            "direction": direction,
            "confidence": round(float(confidence), 1),
            "lstm_price": float(lstm_price[-1]),
            "transformer_price": float(trans_price[-1]),
            "lstm_dir_prob": float(lstm_dir[-1]),
            "transformer_dir_prob": float(trans_dir[-1]),
            "signal_strength": signal_strength,
        }

        self._recent_predictions.append(result)
        if len(self._recent_predictions) > 100:
            self._recent_predictions.pop(0)

        return result

    def update_weights(self, true_price: float, true_direction: int) -> None:
        """Dynamically adjust model weights based on recent errors.

        Args:
            true_price: Actual closing price.
            true_direction: 1 for up, 0 for down.
        """
        if not self._recent_predictions:
            return
        last = self._recent_predictions[-1]

        lstm_price_err = abs(last["lstm_price"] - true_price) / max(true_price, 0.01)
        trans_price_err = abs(last["transformer_price"] - true_price) / max(true_price, 0.01)
        lstm_dir_err = abs(last["lstm_dir_prob"] - true_direction)
        trans_dir_err = abs(last["transformer_dir_prob"] - true_direction)

        lstm_total_err = lstm_price_err * 0.6 + lstm_dir_err * 0.4
        trans_total_err = trans_price_err * 0.6 + trans_dir_err * 0.4

        self._lstm_errors.append(lstm_total_err)
        self._transformer_errors.append(trans_total_err)

        for err_list in (self._lstm_errors, self._transformer_errors):
            if len(err_list) > 50:
                err_list.pop(0)

        if len(self._lstm_errors) >= 5:
            avg_lstm_err = np.mean(self._lstm_errors[-10:]) + 1e-6
            avg_trans_err = np.mean(self._transformer_errors[-10:]) + 1e-6
            self.transformer_weight = avg_lstm_err / (avg_lstm_err + avg_trans_err)
            self.lstm_weight = 1.0 - self.transformer_weight
            logger.debug(f"Updated weights: LSTM={self.lstm_weight:.3f}, Transformer={self.transformer_weight:.3f}")

    def save_all(self, ticker: str) -> None:
        """Save both models and weights for a ticker."""
        base = self.models_dir / ticker
        base.mkdir(parents=True, exist_ok=True)
        if self.lstm:
            self.lstm.save(str(base / "lstm_model.pt"))
        if self.transformer:
            self.transformer.save(str(base / "transformer_model.pt"))
        weight_data = {
            "lstm_weight": self.lstm_weight,
            "transformer_weight": self.transformer_weight,
        }
        with open(base / "ensemble_weights.json", "w") as f:
            json.dump(weight_data, f)
        logger.info(f"Ensemble saved for {ticker}")

    def load_all(self, ticker: str) -> bool:
        """Load both models and weights for a ticker."""
        from src.models.lstm_model import LSTMModel
        from src.models.transformer_model import TransformerModel

        base = self.models_dir / ticker
        lstm_path = base / "lstm_model.pt"
        trans_path = base / "transformer_model.pt"

        if not lstm_path.exists() and not trans_path.exists():
            return False

        lstm_loaded = None
        trans_loaded = None
        if lstm_path.exists():
            lstm_loaded = LSTMModel.load(str(lstm_path))
        if trans_path.exists():
            trans_loaded = TransformerModel.load(str(trans_path))

        if lstm_loaded:
            self.lstm = lstm_loaded
        if trans_loaded:
            self.transformer = trans_loaded

        weights_path = base / "ensemble_weights.json"
        if weights_path.exists():
            try:
                with open(weights_path) as f:
                    data = json.load(f)
                self.lstm_weight = data.get("lstm_weight", 0.5)
                self.transformer_weight = data.get("transformer_weight", 0.5)
            except Exception:
                pass

        success = lstm_loaded is not None or trans_loaded is not None
        if success:
            logger.info(f"Ensemble loaded for {ticker}")
        return success

    @property
    def current_weights(self) -> dict:
        return {
            "lstm_weight": round(self.lstm_weight, 3),
            "transformer_weight": round(self.transformer_weight, 3),
        }
