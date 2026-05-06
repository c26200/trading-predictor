"""Feature engineering pipeline: technical indicators, scaling, and sequence windowing."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

logger = logging.getLogger(__name__)


class FeaturePreprocessor:
    """Full feature engineering pipeline for OHLCV data."""

    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizon: int = 5,
        scaler_type: str = "minmax",
        scaler_path: Optional[str] = None,
    ) -> None:
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler_type = scaler_type
        self.scaler: Optional[MinMaxScaler | RobustScaler] = None
        self.scaler_path = scaler_path
        self.price_scaler: Optional[MinMaxScaler] = None
        self.feature_names_: list[str] = []
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Engineer features and transform data. Returns (X, y) arrays."""
        df = self._ensure_columns(df)
        df = self._add_technical_indicators(df)
        df = self._add_lag_features(df)
        df = self._add_return_features(df)
        df = self._add_volatility_features(df)
        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()

        if df.empty:
            raise ValueError("DataFrame empty after feature engineering — not enough data.")

        # Store price for unscaling later
        self.price_col_idx_ = df.columns.get_loc("close")

        feature_arr = df.values.astype(np.float64)

        # Fit scalers
        if self.scaler_type == "minmax":
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            self.scaler = RobustScaler()
        self.price_scaler = MinMaxScaler(feature_range=(-1, 1))

        scaled = self.scaler.fit_transform(feature_arr)
        price_scaled = self.price_scaler.fit_transform(feature_arr[:, self.price_col_idx_].reshape(-1, 1))

        self.feature_names_ = list(df.columns)
        self._fitted = True

        X, y = self._create_sequences(scaled, price_scaled.flatten())
        return X, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted scaler. Returns latest sequence."""
        if not self._fitted or self.scaler is None:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")

        df = self._ensure_columns(df)
        df = self._add_technical_indicators(df)
        df = self._add_lag_features(df)
        df = self._add_return_features(df)
        df = self._add_volatility_features(df)
        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()

        # Align columns with fitted features
        for col in self.feature_names_:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.feature_names_]

        arr = df.values.astype(np.float64)
        scaled = self.scaler.transform(arr)
        if len(scaled) < self.sequence_length:
            # Pad by repeating first row
            pad = np.repeat(scaled[:1], self.sequence_length - len(scaled), axis=0)
            scaled = np.concatenate([pad, scaled], axis=0)
        return scaled[-self.sequence_length:]

    def inverse_transform_price(self, scaled_price: np.ndarray) -> np.ndarray:
        """Inverse transform price predictions back to original scale."""
        if self.price_scaler is None:
            return scaled_price
        return self.price_scaler.inverse_transform(scaled_price.reshape(-1, 1)).flatten()

    def save_scalers(self, path: str) -> None:
        """Save fitted scaler state to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "price_scaler": self.price_scaler,
                "feature_names": self.feature_names_,
            }, f)
        logger.info(f"Scalers saved to {path}")

    def load_scalers(self, path: str) -> bool:
        """Load scaler state from disk. Returns True if successful."""
        p = Path(path)
        if not p.exists():
            logger.warning(f"Scaler file not found: {path}")
            return False
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
            self.scaler = data["scaler"]
            self.price_scaler = data["price_scaler"]
            self.feature_names_ = data.get("feature_names", [])
            self._fitted = True
            logger.info(f"Scalers loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load scalers: {e}")
            return False

    # ── Private helpers ──────────────────────────────────────────

    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure OHLCV columns exist with correct names."""
        rename = {
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
            "Adj Close": "adj_close",
        }
        df = df.copy()
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                # Derive from close if missing
                if "close" in df.columns and col != "volume":
                    df[col] = df["close"]
                elif col == "volume":
                    df[col] = 1_000_000
        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI, MACD, Bollinger Bands, ATR, EMA, Stochastic, OBV, VWAP."""
        if not TA_AVAILABLE:
            return self._manual_indicators(df)

        try:
            close = df["close"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            volume = df["volume"].astype(float)

            df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
            macd = ta.trend.MACD(close)
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()
            df["macd_diff"] = macd.macd_diff()
            bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
            df["bb_upper"] = bb.bollinger_hband()
            df["bb_middle"] = bb.bollinger_mavg()
            df["bb_lower"] = bb.bollinger_lband()
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
            df["atr"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
            df["ema_9"] = ta.trend.EMAIndicator(close, window=9).ema_indicator()
            df["ema_21"] = ta.trend.EMAIndicator(close, window=21).ema_indicator()
            df["ema_50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()
            stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
            df["stoch_k"] = stoch.stoch()
            df["stoch_d"] = stoch.stoch_signal()
            df["obv"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()

            try:
                df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
            except Exception:
                df["vwap"] = df["close"]

        except Exception as e:
            logger.warning(f"TA indicators failed: {e}. Using manual fallback.")
            df = self._manual_indicators(df)

        return df

    def _manual_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Manual indicator calculations as fallback when 'ta' library unavailable."""
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)

        # RSI 14
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_diff"] = df["macd"] - df["macd_signal"]

        # Bollinger Bands
        df["bb_middle"] = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * std20
        df["bb_lower"] = df["bb_middle"] - 2 * std20
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # ATR 14
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

        # EMA
        df["ema_9"] = close.ewm(span=9).mean()
        df["ema_21"] = close.ewm(span=21).mean()
        df["ema_50"] = close.ewm(span=50).mean()

        # Stochastic %K, %D
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        df["stoch_k"] = 100 * (close - low14) / (high14 - low14 + 1e-9)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # OBV
        df["obv"] = (volume * ((close > close.shift(1)).astype(int) * 2 - 1)).cumsum()

        # VWAP
        df["vwap"] = (close * volume).cumsum() / volume.cumsum()

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add t-1 through t-10 lag features for close price."""
        for lag in range(1, 11):
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
        return df

    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add returns and log-returns."""
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling volatility windows (5, 10, 20)."""
        ret = df.get("returns", df["close"].pct_change())
        for window in [5, 10, 20]:
            df[f"volatility_{window}"] = ret.rolling(window).std()
        return df

    def _create_sequences(
        self, data: np.ndarray, price_target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences for training."""
        X, y_price, y_dir = [], [], []
        seq_len = self.sequence_length
        horizon = self.prediction_horizon

        for i in range(len(data) - seq_len - horizon):
            X.append(data[i : i + seq_len])
            future_price = price_target[i + seq_len + horizon - 1]
            current_price = price_target[i + seq_len - 1]
            y_price.append(future_price)
            y_dir.append(1 if future_price > current_price else 0)

        if not X:
            raise ValueError(f"Not enough data for sequences: {len(data)} rows, need {seq_len + horizon}")

        return np.array(X), np.column_stack([np.array(y_price), np.array(y_dir)])
