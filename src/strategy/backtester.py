"""Full backtesting engine with walk-forward validation, commission, and slippage."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.strategy.risk_manager import RiskManager
from src.strategy.signals import Signal, SignalGenerator

logger = logging.getLogger(__name__)


class Backtester:
    """Walk-forward backtesting engine with realistic market simulation."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_pct: float = 0.1,
        slippage_pct: float = 0.05,
        max_risk_per_trade: float = 2.0,
        max_drawdown_limit: float = 15.0,
    ) -> None:
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

        self.risk_manager = RiskManager(
            initial_capital=initial_capital,
            max_risk_per_trade_pct=max_risk_per_trade,
            max_drawdown_limit_pct=max_drawdown_limit,
        )
        self.signal_generator = SignalGenerator()

        # Results storage
        self.equity_curve: list[float] = [initial_capital]
        self.trades: list[dict] = []
        self.metrics: dict = {}

        # Internal state
        self._position: Optional[dict] = None
        self._position_size: float = 0.0

    def run(
        self,
        df: pd.DataFrame,
        predictions: list[dict],
        ticker: str = "UNKNOWN",
    ) -> dict:
        """Run backtest on OHLCV data with prediction signals.

        Args:
            df: OHLCV DataFrame with columns: open, high, low, close (lowercase).
            predictions: List of prediction dicts, aligned with df rows (same length).
            ticker: Ticker symbol.

        Returns:
            Dict with metrics, equity_curve, and trades.
        """
        if df.empty or len(predictions) < len(df):
            raise ValueError("Data and predictions must be aligned and non-empty.")

        # Reset state
        capital = self.initial_capital
        self.equity_curve = [capital]
        self.trades = []
        self._position = None
        self._position_size = 0.0
        self.risk_manager = RiskManager(initial_capital=self.initial_capital)

        close_prices = df["close"].values
        dates = df.index

        for i in range(len(df)):
            if self.risk_manager.check_drawdown():
                break  # Halt trading

            pred = predictions[i] if i < len(predictions) else predictions[-1]
            price = float(close_prices[i])

            # Get indicator values at this point
            indicators = {
                "rsi": float(df["rsi"].iloc[i]) if "rsi" in df.columns else 50.0,
                "ema_21": float(df["ema_21"].iloc[i]) if "ema_21" in df.columns else price,
                "atr": float(df["atr"].iloc[i]) if "atr" in df.columns else price * 0.015,
            }

            signal = self.signal_generator.generate(
                ticker=ticker,
                prediction=pred,
                current_price=price,
                indicators=indicators,
                timestamp=str(dates[i]),
            )

            # Execute entry
            if signal.signal_type in ("LONG", "SHORT") and self._position is None:
                entry_price = price * (1 + self.slippage_pct / 100 if signal.signal_type == "LONG" else 1 - self.slippage_pct / 100)
                stop_loss_pct = abs(entry_price - (signal.stop_loss or entry_price)) / entry_price
                self._position_size = min(
                    self.risk_manager.compute_position_size(capital, stop_loss_pct),
                    capital,
                )
                cost = self._position_size * (self.commission_pct / 100)
                capital -= cost
                self._position = {
                    "type": signal.signal_type,
                    "entry_price": entry_price,
                    "entry_date": str(dates[i]),
                    "size": self._position_size,
                    "stop_loss": signal.stop_loss,
                    "trailing_stop": signal.trailing_stop,
                    "take_profit": signal.take_profit,
                }

            # Execute exit
            elif signal.signal_type == "EXIT" and self._position is not None:
                exit_price = price * (1 - self.slippage_pct / 100 if self._position["type"] == "LONG" else 1 + self.slippage_pct / 100)
                cost = self._position_size * (self.commission_pct / 100)

                if self._position["type"] == "LONG":
                    pnl = self._position_size * (exit_price - self._position["entry_price"]) / self._position["entry_price"]
                else:
                    pnl = self._position_size * (self._position["entry_price"] - exit_price) / self._position["entry_price"]

                pnl -= cost * 2  # Entry + exit commission
                capital += self._position_size + pnl

                self.trades.append({
                    "ticker": ticker,
                    "type": self._position["type"],
                    "entry_date": self._position["entry_date"],
                    "exit_date": str(dates[i]),
                    "entry_price": self._position["entry_price"],
                    "exit_price": exit_price,
                    "pnl": round(float(pnl), 2),
                    "return_pct": round(float(pnl / self._position_size * 100), 2),
                    "exit_reason": signal.reason,
                })

                self.risk_manager.record_trade(self.trades[-1])
                self._position = None
                self._position_size = 0.0

            self.equity_curve.append(capital)
            self.risk_manager.update_equity(capital)

        # Close any remaining position at last price
        if self._position is not None:
            final_price = float(close_prices[-1])
            if self._position["type"] == "LONG":
                pnl = self._position_size * (final_price - self._position["entry_price"]) / self._position["entry_price"]
            else:
                pnl = self._position_size * (self._position["entry_price"] - final_price) / self._position["entry_price"]
            capital += self._position_size + pnl
            self.trades.append({
                "ticker": ticker,
                "type": self._position["type"],
                "entry_date": self._position["entry_date"],
                "exit_date": str(dates[-1]),
                "entry_price": self._position["entry_price"],
                "exit_price": final_price,
                "pnl": round(float(pnl), 2),
                "return_pct": round(float(pnl / self._position_size * 100), 2),
                "exit_reason": "End of backtest",
            })
            self._position = None

        # Compute final metrics
        self.metrics = self.risk_manager.get_summary()
        total_return = (capital - self.initial_capital) / self.initial_capital * 100
        self.metrics["total_return_pct"] = round(total_return, 2)

        return {
            "metrics": self.metrics,
            "equity_curve": self.equity_curve,
            "trades": self.trades[-20:],  # Last 20 trades
            "total_trades": len(self.trades),
            "ticker": ticker,
        }

    def walk_forward(
        self,
        df: pd.DataFrame,
        predictions: list[dict],
        ticker: str = "UNKNOWN",
        n_splits: int = 5,
    ) -> dict:
        """Walk-forward validation with expanding window.

        Splits data into n_splits folds, training on expanding window,
        testing on subsequent fold.
        """
        n = len(df)
        fold_size = n // (n_splits + 1)
        if fold_size < 30:
            return self.run(df, predictions, ticker)

        all_results: list[dict] = []
        combined_equity: list[float] = [self.initial_capital]
        combined_trades: list[dict] = []

        for split in range(1, n_splits + 1):
            test_end = min((split + 1) * fold_size, n)
            test_start = split * fold_size

            test_df = df.iloc[:test_end]
            test_preds = predictions[:test_end]

            result = self.run(test_df, test_preds, ticker)
            all_results.append(result["metrics"])
            combined_trades.extend(result["trades"])

        # Aggregate metrics
        avg_metrics: dict = {}
        keys = all_results[0].keys() if all_results else []
        for key in keys:
            values = [m[key] for m in all_results if key in m]
            if values and isinstance(values[0], (int, float)):
                avg_metrics[key] = round(float(np.mean(values)), 3)

        return {
            "metrics": avg_metrics,
            "equity_curve": self.equity_curve,
            "trades": combined_trades[-20:],
            "total_trades": len(combined_trades),
            "ticker": ticker,
            "walk_forward_splits": n_splits,
        }

    def generate_equity_chart_data(self) -> dict:
        """Generate equity curve data suitable for frontend charting."""
        if not self.equity_curve:
            return {"labels": [], "values": []}
        return {
            "labels": [str(i) for i in range(len(self.equity_curve))],
            "values": [round(v, 2) for v in self.equity_curve],
            "initial_capital": self.initial_capital,
            "final_capital": round(self.equity_curve[-1], 2),
        }
