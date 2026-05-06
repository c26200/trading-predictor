"""Data fetcher: OHLCV from yfinance with synthetic fallback and streaming support."""

from __future__ import annotations

import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_TICKERS = ["AAPL", "TSLA", "BTC-USD", "ETH-USD", "SPY", "NVDA"]


class DataFetcher:
    """Fetches OHLCV data from Yahoo Finance with synthetic fallback."""

    def __init__(
        self,
        tickers: Optional[list[str]] = None,
        lookback_days: int = 730,
        interval: str = "1d",
    ) -> None:
        self.tickers = tickers or DEFAULT_TICKERS
        self.lookback_days = lookback_days
        self.interval = interval
        self._cache: dict[str, pd.DataFrame] = {}
        self._simulation_state: dict[str, dict] = {}

    def fetch_historical(self, ticker: str) -> pd.DataFrame:
        """Fetch OHLCV data for a ticker. Falls back to synthetic on failure."""
        try:
            df = yf.download(
                ticker,
                period=f"{self.lookback_days}d",
                interval=self.interval,
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                raise ValueError(f"No data returned for {ticker}")
            df = self._clean_dataframe(df)
            self._cache[ticker] = df.copy()
            logger.info(f"Fetched {len(df)} rows for {ticker}")
            return df
        except Exception as e:
            logger.warning(f"yfinance failed for {ticker}: {e}. Using synthetic data.")
            return self._generate_synthetic(ticker)

    def fetch_all(self) -> dict[str, pd.DataFrame]:
        """Fetch historical data for all tickers."""
        results: dict[str, pd.DataFrame] = {}
        for ticker in self.tickers:
            results[ticker] = self.fetch_historical(ticker)
        return results

    def get_latest(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get the latest data for a ticker (cached + new)."""
        if ticker in self._cache and not self._cache[ticker].empty:
            return self._cache[ticker].tail(1)
        return None

    def stream_tick(self, ticker: str) -> dict:
        """Simulate a real-time data tick. Returns a dict with price info."""
        if ticker not in self._simulation_state:
            df = self._cache.get(ticker)
            if df is not None and not df.empty:
                last_row = df.iloc[-1]
                base_price = float(last_row["Close"])
                mu = float(df["Close"].pct_change().mean()) if len(df) > 1 else 0.0001
                sigma = float(df["Close"].pct_change().std()) if len(df) > 1 else 0.02
            else:
                base_price = self._default_price(ticker)
                mu = 0.0001
                sigma = 0.02
            self._simulation_state[ticker] = {
                "price": base_price,
                "mu": mu,
                "sigma": sigma,
                "timestamp": datetime.now(),
            }

        state = self._simulation_state[ticker]
        # Geometric Brownian motion step
        drift = state["mu"] - 0.5 * state["sigma"] ** 2
        shock = state["sigma"] * np.random.normal(0, 1)
        new_price = state["price"] * np.exp(drift + shock)
        state["price"] = new_price
        state["timestamp"] = datetime.now()

        change_pct = (new_price / max(state.get("prev_price", new_price), 0.01) - 1) * 100 if "prev_price" in state else 0.0
        state["prev_price"] = float(new_price)

        return {
            "ticker": ticker,
            "price": round(float(new_price), 2),
            "change_pct": round(float(change_pct), 2),
            "timestamp": state["timestamp"].isoformat(),
            "volume": int(abs(np.random.normal(5_000_000, 2_000_000))),
            "simulated": True,
        }

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column MultiIndex and rename to simple names."""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        col_map = {
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
            "open": "open", "high": "high", "low": "low",
            "close": "close", "volume": "volume",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df = df.dropna()
        return df

    def _generate_synthetic(self, ticker: str) -> pd.DataFrame:
        """Generate synthetic OHLCV data using Geometric Brownian Motion."""
        base = self._default_price(ticker)
        dates = pd.date_range(
            end=datetime.now(),
            periods=self.lookback_days,
            freq="B",
        )
        returns = np.random.normal(0.0003, 0.015, len(dates))
        prices = base * np.exp(np.cumsum(returns))

        data: dict[str, list] = []
        for i, (d, p) in enumerate(zip(dates, prices)):
            daily_vol = p * 0.015
            open_p = prices[i - 1] if i > 0 else p * 0.999
            close_p = p
            high_p = max(open_p, close_p) + abs(np.random.normal(0, daily_vol * 0.5))
            low_p = min(open_p, close_p) - abs(np.random.normal(0, daily_vol * 0.5))
            vol = int(abs(np.random.normal(5_000_000, 2_000_000)))
            data.append({
                "open": round(float(open_p), 2),
                "high": round(float(high_p), 2),
                "low": round(float(low_p), 2),
                "close": round(float(close_p), 2),
                "volume": vol,
            })

        df = pd.DataFrame(data, index=dates)
        self._cache[ticker] = df.copy()
        logger.info(f"Generated {len(df)} synthetic rows for {ticker}")
        return df

    def _default_price(self, ticker: str) -> float:
        """Return default base price for a ticker."""
        defaults: dict[str, float] = {
            "AAPL": 180.0, "TSLA": 250.0, "BTC-USD": 65000.0,
            "ETH-USD": 3500.0, "SPY": 520.0, "NVDA": 900.0,
        }
        return defaults.get(ticker, 100.0)

    @staticmethod
    def get_default_tickers() -> list[str]:
        return list(DEFAULT_TICKERS)
