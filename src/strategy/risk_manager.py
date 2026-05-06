"""Portfolio risk management: position sizing, drawdown guards, and ratio calculations."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RiskManager:
    """Manages portfolio risk, position sizing, and performance metrics."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        max_risk_per_trade_pct: float = 2.0,
        max_drawdown_limit_pct: float = 15.0,
        max_correlation_exposure: float = 0.70,
    ) -> None:
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.max_drawdown_limit_pct = max_drawdown_limit_pct
        self.max_correlation_exposure = max_correlation_exposure

        self._equity_curve: list[float] = [initial_capital]
        self._trades: list[dict] = []
        self._peak_capital = initial_capital
        self._trading_halted = False
        self._positions: dict[str, dict] = {}

    def kelly_position_size(
        self, win_probability: float, avg_win: float, avg_loss: float
    ) -> float:
        """Compute Kelly Criterion position size fraction.

        Args:
            win_probability: Probability of winning (0-1).
            avg_win: Average winning trade return (positive).
            avg_loss: Average losing trade return (positive magnitude).
        """
        if avg_loss == 0:
            return 0.0
        b = avg_win / avg_loss  # Win/loss ratio
        kelly = (win_probability * b - (1 - win_probability)) / b
        # Use half-Kelly for safety
        return max(0.0, min(kelly * 0.5, 0.25))

    def compute_position_size(self, capital: float, stop_loss_pct: float) -> float:
        """Compute position size using fixed fractional method.

        Args:
            capital: Current portfolio capital.
            stop_loss_pct: Distance to stop-loss as percentage (0.02 = 2%).
        """
        max_risk_amount = capital * (self.max_risk_per_trade_pct / 100.0)
        if stop_loss_pct > 0:
            position_size = max_risk_amount / stop_loss_pct
        else:
            position_size = capital * 0.1
        # Cap at 25% of capital per position
        return min(position_size, capital * 0.25)

    def check_drawdown(self) -> bool:
        """Check if max drawdown limit is exceeded. Returns True if trading should halt."""
        if not self._equity_curve:
            return False
        current_equity = self._equity_curve[-1]
        self._peak_capital = max(self._peak_capital, current_equity)
        drawdown = (self._peak_capital - current_equity) / self._peak_capital * 100

        if drawdown > self.max_drawdown_limit_pct:
            if not self._trading_halted:
                logger.warning(
                    f"Max drawdown limit ({self.max_drawdown_limit_pct}%) exceeded: {drawdown:.1f}%. Halting trading."
                )
                self._trading_halted = True
            return True
        return False

    def update_equity(self, new_capital: float) -> None:
        """Update equity curve and current capital."""
        self.current_capital = new_capital
        self._equity_curve.append(new_capital)
        self._peak_capital = max(self._peak_capital, new_capital)
        # Resume trading if capital recovers above drawdown limit
        if self._trading_halted:
            dd = (self._peak_capital - new_capital) / self._peak_capital * 100
            if dd < self.max_drawdown_limit_pct:
                self._trading_halted = False
                logger.info("Drawdown recovered below limit. Trading resumed.")

    def record_trade(self, trade: dict) -> None:
        """Record a completed trade."""
        self._trades.append(trade)

    def get_correlation_matrix(self, returns_dict: dict[str, pd.Series]) -> pd.DataFrame:
        """Compute correlation matrix of returns across tickers.

        Args:
            returns_dict: Dict mapping ticker -> daily returns Series.
        """
        if len(returns_dict) < 2:
            return pd.DataFrame([[1.0]])
        df = pd.DataFrame(returns_dict)
        return df.corr()

    def check_correlation_limits(
        self, ticker: str, returns_dict: dict[str, pd.Series]
    ) -> bool:
        """Check if adding exposure to ticker exceeds correlation limits.

        Returns True if exposure is allowed.
        """
        if ticker not in returns_dict:
            return True
        corr = self.get_correlation_matrix(returns_dict)
        ticker_corrs = corr[ticker].drop(ticker)
        if ticker_corrs.empty:
            return True
        return ticker_corrs.max() <= self.max_correlation_exposure

    # ── Performance Metrics ─────────────────────────────────────

    def sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Compute Sharpe ratio from equity curve."""
        if len(self._equity_curve) < 2:
            return 0.0
        equity = pd.Series(self._equity_curve)
        daily_returns = equity.pct_change().dropna()
        if len(daily_returns) < 2 or daily_returns.std() == 0:
            return 0.0
        excess_returns = daily_returns - risk_free_rate / 252
        return float(np.sqrt(252) * excess_returns.mean() / daily_returns.std())

    def sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Compute Sortino ratio (uses downside deviation only)."""
        if len(self._equity_curve) < 2:
            return 0.0
        equity = pd.Series(self._equity_curve)
        daily_returns = equity.pct_change().dropna()
        if len(daily_returns) < 2:
            return 0.0
        downside = daily_returns[daily_returns < 0]
        if len(downside) < 2 or downside.std() == 0:
            return 0.0
        excess = daily_returns - risk_free_rate / 252
        return float(np.sqrt(252) * excess.mean() / downside.std())

    def calmar_ratio(self) -> float:
        """Compute Calmar ratio: annualized return / max drawdown."""
        if len(self._equity_curve) < 20:
            return 0.0
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        # Annualized: (1 + total_return)^(252/n_days) - 1
        n_days = len(self._equity_curve)
        if n_days < 2:
            return 0.0
        annualized_return = (1 + total_return) ** (252 / n_days) - 1
        max_dd = self.max_drawdown_pct()
        if max_dd < 0.01:
            return 0.0
        return float(annualized_return / (max_dd / 100))

    def max_drawdown_pct(self) -> float:
        """Compute maximum drawdown percentage."""
        if not self._equity_curve:
            return 0.0
        curve = np.array(self._equity_curve)
        peak = np.maximum.accumulate(curve)
        drawdowns = (peak - curve) / peak * 100
        return float(np.max(drawdowns))

    def win_rate(self) -> float:
        """Compute win rate from recorded trades."""
        if not self._trades:
            return 0.0
        wins = sum(1 for t in self._trades if t.get("pnl", 0) > 0)
        return wins / len(self._trades) * 100

    def profit_factor(self) -> float:
        """Compute profit factor: gross profit / gross loss."""
        gross_profit = sum(t["pnl"] for t in self._trades if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t["pnl"] for t in self._trades if t.get("pnl", 0) < 0))
        if gross_loss < 0.01:
            return gross_profit if gross_profit > 0 else float("inf")
        return round(gross_profit / gross_loss, 2)

    def total_return_pct(self) -> float:
        """Total return percentage."""
        return round((self.current_capital - self.initial_capital) / self.initial_capital * 100, 2)

    def is_trading_halted(self) -> bool:
        return self._trading_halted

    def get_summary(self) -> dict:
        """Return comprehensive portfolio summary."""
        return {
            "initial_capital": self.initial_capital,
            "current_capital": round(self.current_capital, 2),
            "total_return_pct": self.total_return_pct(),
            "max_drawdown_pct": round(self.max_drawdown_pct(), 2),
            "sharpe_ratio": round(self.sharpe_ratio(), 3),
            "sortino_ratio": round(self.sortino_ratio(), 3),
            "calmar_ratio": round(self.calmar_ratio(), 3),
            "win_rate_pct": round(self.win_rate(), 1),
            "profit_factor": self.profit_factor(),
            "total_trades": len(self._trades),
            "trading_halted": self._trading_halted,
        }
