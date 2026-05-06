"""Entry/Exit signal generator based on ensemble predictions and technical indicators."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Trading signal data container."""

    ticker: str
    signal_type: str  # LONG, SHORT, EXIT, HOLD
    price: float
    confidence: float
    direction: str  # UP, DOWN, NEUTRAL
    strength: int  # 1-5 stars
    reason: str
    timestamp: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None


class SignalGenerator:
    """Generates trading signals from ensemble predictions and indicators."""

    def __init__(
        self,
        confidence_threshold: float = 75.0,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        stop_loss_atr: float = 2.0,
        take_profit_atr: float = 4.0,
        trailing_stop_atr: float = 1.5,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
        self.trailing_stop_atr = trailing_stop_atr
        self._active_position: Optional[Signal] = None
        self._signal_history: list[Signal] = []

    def generate(
        self,
        ticker: str,
        prediction: dict,
        current_price: float,
        indicators: Optional[dict] = None,
        timestamp: str = "",
    ) -> Signal:
        """Generate trading signal from prediction and current market data.

        Args:
            ticker: Asset ticker symbol.
            prediction: Ensemble prediction dict (from EnsemblePredictor.predict).
            current_price: Current market price.
            indicators: Dict with rsi, ema_21, atr values.
            timestamp: ISO format timestamp string.
        """
        confidence = prediction["confidence"]
        direction = prediction["direction"]
        strength = prediction["signal_strength"]
        indicators = indicators or {}

        rsi = indicators.get("rsi", 50.0)
        ema_21 = indicators.get("ema_21", current_price)
        atr = indicators.get("atr", current_price * 0.01)
        price_above_ema = current_price > ema_21

        # Active position management
        if self._active_position is not None:
            exit_signal = self._check_exit_conditions(
                current_price, atr, self._active_position
            )
            if exit_signal:
                self._active_position = None
                return exit_signal
            # Update trailing stop
            if self._active_position.signal_type == "LONG":
                new_trail = current_price - self.trailing_stop_atr * atr
                if self._active_position.trailing_stop is not None:
                    self._active_position.trailing_stop = max(
                        self._active_position.trailing_stop, new_trail
                    )
            elif self._active_position.signal_type == "SHORT":
                new_trail = current_price + self.trailing_stop_atr * atr
                if self._active_position.trailing_stop is not None:
                    self._active_position.trailing_stop = min(
                        self._active_position.trailing_stop, new_trail
                    )

        # LONG signal conditions
        if (
            confidence >= self.confidence_threshold
            and direction == "UP"
            and rsi < self.rsi_overbought
            and price_above_ema
            and self._active_position is None
        ):
            signal = Signal(
                ticker=ticker,
                signal_type="LONG",
                price=current_price,
                confidence=confidence,
                direction=direction,
                strength=strength,
                reason=f"High confidence UP ({confidence:.0f}%), RSI={rsi:.0f}, Price > EMA21",
                timestamp=timestamp,
                stop_loss=current_price - self.stop_loss_atr * atr,
                take_profit=current_price + self.take_profit_atr * atr,
                trailing_stop=current_price - self.trailing_stop_atr * atr,
            )
            self._active_position = signal
            self._signal_history.append(signal)
            if len(self._signal_history) > 50:
                self._signal_history.pop(0)
            logger.info(f"LONG signal: {ticker} @ ${current_price:.2f} | Confidence: {confidence:.0f}%")
            return signal

        # SHORT signal conditions
        if (
            confidence >= self.confidence_threshold
            and direction == "DOWN"
            and rsi > self.rsi_oversold
            and not price_above_ema
            and self._active_position is None
        ):
            signal = Signal(
                ticker=ticker,
                signal_type="SHORT",
                price=current_price,
                confidence=confidence,
                direction=direction,
                strength=strength,
                reason=f"High confidence DOWN ({confidence:.0f}%), RSI={rsi:.0f}, Price < EMA21",
                timestamp=timestamp,
                stop_loss=current_price + self.stop_loss_atr * atr,
                take_profit=current_price - self.take_profit_atr * atr,
                trailing_stop=current_price + self.trailing_stop_atr * atr,
            )
            self._active_position = signal
            self._signal_history.append(signal)
            if len(self._signal_history) > 50:
                self._signal_history.pop(0)
            logger.info(f"SHORT signal: {ticker} @ ${current_price:.2f} | Confidence: {confidence:.0f}%")
            return signal

        # HOLD
        return Signal(
            ticker=ticker,
            signal_type="HOLD",
            price=current_price,
            confidence=confidence,
            direction=direction,
            strength=strength,
            reason="No entry criteria met",
            timestamp=timestamp,
        )

    def _check_exit_conditions(
        self, current_price: float, atr: float, position: Signal
    ) -> Optional[Signal]:
        """Check if exit conditions are met for active position."""
        if position.signal_type == "LONG":
            if position.stop_loss and current_price <= position.stop_loss:
                return Signal(
                    ticker=position.ticker,
                    signal_type="EXIT",
                    price=current_price,
                    confidence=0,
                    direction="DOWN",
                    strength=1,
                    reason=f"Stop-loss hit: ${position.stop_loss:.2f}",
                    timestamp="",
                )
            if position.take_profit and current_price >= position.take_profit:
                return Signal(
                    ticker=position.ticker,
                    signal_type="EXIT",
                    price=current_price,
                    confidence=0,
                    direction="UP",
                    strength=1,
                    reason=f"Take-profit hit: ${position.take_profit:.2f}",
                    timestamp="",
                )
            if position.trailing_stop and current_price <= position.trailing_stop:
                return Signal(
                    ticker=position.ticker,
                    signal_type="EXIT",
                    price=current_price,
                    confidence=0,
                    direction="DOWN",
                    strength=1,
                    reason=f"Trailing stop hit: ${position.trailing_stop:.2f}",
                    timestamp="",
                )
        elif position.signal_type == "SHORT":
            if position.stop_loss and current_price >= position.stop_loss:
                return Signal(
                    ticker=position.ticker,
                    signal_type="EXIT",
                    price=current_price,
                    confidence=0,
                    direction="UP",
                    strength=1,
                    reason=f"Stop-loss hit: ${position.stop_loss:.2f}",
                    timestamp="",
                )
            if position.take_profit and current_price <= position.take_profit:
                return Signal(
                    ticker=position.ticker,
                    signal_type="EXIT",
                    price=current_price,
                    confidence=0,
                    direction="DOWN",
                    strength=1,
                    reason=f"Take-profit hit: ${position.take_profit:.2f}",
                    timestamp="",
                )
        return None

    def get_history(self, limit: int = 50) -> list[dict]:
        """Return recent signal history as dicts."""
        return [
            {
                "ticker": s.ticker,
                "signal_type": s.signal_type,
                "price": s.price,
                "confidence": s.confidence,
                "direction": s.direction,
                "strength": s.strength,
                "reason": s.reason,
                "timestamp": s.timestamp,
            }
            for s in self._signal_history[-limit:]
        ]

    def reset(self) -> None:
        """Reset active position and history."""
        self._active_position = None
        self._signal_history.clear()
