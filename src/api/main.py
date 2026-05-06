"""FastAPI application: REST + WebSocket endpoints for trading dashboard."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("api")

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False

# ── App Setup ───────────────────────────────────────────────────

if not FASTAPI_AVAILABLE:
    raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")

app = FastAPI(
    title="Trading Predictor API",
    description="Algorithmic Trading Prediction System with Deep Learning",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic Models ─────────────────────────────────────────────


class PredictionResponse(BaseModel):
    ticker: str
    predicted_price: float
    current_price: float
    direction: str
    confidence: float
    signal_strength: int
    lstm_weight: float
    transformer_weight: float
    timestamp: str


class SignalResponse(BaseModel):
    ticker: str
    signal_type: str
    price: float
    confidence: float
    direction: str
    strength: int
    reason: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    timestamp: str


class BacktestResponse(BaseModel):
    ticker: str
    metrics: dict
    equity_curve: dict
    trades: list[dict]
    total_trades: int


class PortfolioResponse(BaseModel):
    initial_capital: float
    current_capital: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate_pct: float
    profit_factor: float
    total_trades: int
    trading_halted: bool


# ── Lazy imports for optional heavy deps ────────────────────────


def _import_signal_generator():
    from src.strategy.signals import SignalGenerator
    return SignalGenerator


def _import_backtester():
    from src.strategy.backtester import Backtester
    return Backtester


# ── Global State ────────────────────────────────────────────────


class AppState:
    """Singleton application state holding all live components."""

    def __init__(self) -> None:
        self.simulation_mode: bool = True
        self.tickers: list[str] = ["AAPL", "TSLA", "BTC-USD", "ETH-USD", "SPY", "NVDA"]
        self.ensembles: dict = {}
        self.preprocessors: dict = {}
        self.signal_generators: dict = {}
        self.risk_manager = None
        self.backtester = None
        self.latest_predictions: dict = {}
        self.latest_signals: dict = {}
        self.signal_history: list[dict] = []
        self.connected_clients: list[WebSocket] = []
        self.last_refresh: dict[str, float] = {}
        self.fetcher = None

        # Initialize components
        self._init_components()

    def _init_components(self) -> None:
        """Lazy init to avoid heavy imports at module level."""
        try:
            from src.data.fetcher import DataFetcher
            from src.models.ensemble import EnsemblePredictor
            from src.strategy.risk_manager import RiskManager
            from src.strategy.backtester import Backtester

            self.fetcher = DataFetcher(tickers=self.tickers)
            self.risk_manager = RiskManager()
            self.backtester = Backtester()

            # Try loading saved models for each ticker
            for ticker in self.tickers:
                try:
                    ensemble = EnsemblePredictor(models_dir="models/saved")
                    if ensemble.load_all(ticker):
                        from src.data.preprocessor import FeaturePreprocessor
                        preprocessor = FeaturePreprocessor()
                        preprocessor.load_scalers(f"models/saved/{ticker}/scaler.pkl")
                        self.preprocessors[ticker] = preprocessor
                        self.ensembles[ticker] = ensemble
                        from src.strategy.signals import SignalGenerator
                        self.signal_generators[ticker] = SignalGenerator()
                        logger.info(f"Loaded saved models for {ticker}")
                    else:
                        logger.info(f"No saved models for {ticker} — using simulation")
                except Exception as e:
                    logger.warning(f"Could not load models for {ticker}: {e}")

            if self.ensembles:
                self.simulation_mode = False
                logger.info(f"Running in LIVE mode with {len(self.ensembles)} tickers")
            else:
                logger.info("Running in SIMULATION mode — no models loaded")

        except Exception as e:
            logger.warning(f"Error initializing components: {e}. Running in simulation mode.")
            self.simulation_mode = True

    def get_or_create_prediction(self, ticker: str) -> dict:
        """Get latest prediction (simulated if no model available)."""
        now = datetime.now().isoformat()

        if ticker in self.ensembles and not self.simulation_mode:
            try:
                preprocessor = self.preprocessors.get(ticker)
                ensemble = self.ensembles.get(ticker)
                if preprocessor and ensemble and self.fetcher:
                    df = self.fetcher.fetch_historical(ticker)
                    seq = preprocessor.transform(df)
                    X = seq[np.newaxis, ...]
                    pred = ensemble.predict(X)
                    current_price = float(df["close"].iloc[-1])

                    # Inverse transform the predicted price from scaled space
                    predicted_price_scaled = np.array([pred["predicted_price"]])
                    real_price = float(preprocessor.inverse_transform_price(predicted_price_scaled)[0])
                    pred["predicted_price"] = round(real_price, 2)

                    return {
                        "ticker": ticker,
                        "predicted_price": round(real_price, 2),
                        "current_price": round(current_price, 2),
                        "direction": pred["direction"],
                        "confidence": pred["confidence"],
                        "signal_strength": pred["signal_strength"],
                        "lstm_weight": ensemble.lstm_weight,
                        "transformer_weight": ensemble.transformer_weight,
                        "timestamp": now,
                    }
            except Exception as e:
                logger.error(f"Prediction error for {ticker}: {e}")

        return self._simulate_prediction(ticker, now)

    def _simulate_prediction(self, ticker: str, timestamp: str) -> dict:
        """Generate simulated prediction."""
        base_prices = {"AAPL": 185, "TSLA": 260, "BTC-USD": 68000, "ETH-USD": 3600, "SPY": 530, "NVDA": 950}
        base = base_prices.get(ticker, 100)
        noise = np.random.normal(0, base * 0.005)
        current = base + noise * 0.5
        predicted = current + np.random.normal(0, base * 0.01)
        direction = "UP" if predicted > current else "DOWN"
        confidence = np.random.uniform(55, 92)

        return {
            "ticker": ticker,
            "predicted_price": round(float(predicted), 2),
            "current_price": round(float(current), 2),
            "direction": direction,
            "confidence": round(float(confidence), 1),
            "signal_strength": int(np.random.randint(1, 6)),
            "lstm_weight": 0.5,
            "transformer_weight": 0.5,
            "timestamp": timestamp,
            "simulated": True,
        }


state = AppState()


# ── REST Endpoints ──────────────────────────────────────────────


@app.get("/api/predict/{ticker}", response_model=PredictionResponse)
async def get_prediction(ticker: str) -> dict:
    """Get latest prediction for a ticker with confidence score."""
    ticker = ticker.upper()
    pred = state.get_or_create_prediction(ticker)
    state.latest_predictions[ticker] = pred
    return pred


@app.get("/api/signals/{ticker}", response_model=SignalResponse)
async def get_signal(ticker: str) -> dict:
    """Get current Entry/Exit signals for a ticker."""
    ticker = ticker.upper()
    pred = state.get_or_create_prediction(ticker)

    SignalGenerator = _import_signal_generator()
    sig_gen = state.signal_generators.get(ticker, SignalGenerator())
    if ticker not in state.signal_generators:
        state.signal_generators[ticker] = sig_gen

    indicators = {"rsi": np.random.uniform(30, 70), "ema_21": pred["current_price"] * 0.99, "atr": pred["current_price"] * 0.015}
    signal = sig_gen.generate(ticker, pred, pred["current_price"], indicators, pred["timestamp"])
    state.latest_signals[ticker] = signal

    result = {
        "ticker": signal.ticker,
        "signal_type": signal.signal_type,
        "price": signal.price,
        "confidence": signal.confidence,
        "direction": signal.direction,
        "strength": signal.strength,
        "reason": signal.reason,
        "stop_loss": signal.stop_loss,
        "take_profit": signal.take_profit,
        "trailing_stop": signal.trailing_stop,
        "timestamp": signal.timestamp,
    }

    # Add to history for non-HOLD signals
    if signal.signal_type != "HOLD":
        state.signal_history.append(result)
        if len(state.signal_history) > 50:
            state.signal_history.pop(0)

    return result


@app.get("/api/backtest/{ticker}")
async def get_backtest(ticker: str) -> dict:
    """Get backtest results for a ticker."""
    ticker = ticker.upper()
    try:
        fetcher = state.fetcher if hasattr(state, "fetcher") else None
        if fetcher is None:
            return _demo_backtest(ticker)

        df = fetcher.fetch_historical(ticker)
        if df.empty:
            return _demo_backtest(ticker)

        # Generate mock predictions for demo
        n = len(df)
        predictions = [
            {
                "predicted_price": float(df["close"].iloc[min(i + 5, n - 1)]),
                "direction": "UP" if i < n - 5 and df["close"].iloc[min(i + 5, n - 1)] > df["close"].iloc[i] else "DOWN",
                "confidence": np.random.uniform(60, 90),
                "signal_strength": np.random.randint(2, 6),
            }
            for i in range(n)
        ]

        Backtester = _import_backtester()
        bt = Backtester()
        result = bt.run(df, predictions, ticker)
        result["equity_curve"] = bt.generate_equity_chart_data()
        return result
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return _demo_backtest(ticker)


def _demo_backtest(ticker: str) -> dict:
    """Demo backtest results."""
    equity = [100000]
    for i in range(250):
        equity.append(equity[-1] * (1 + np.random.normal(0.0002, 0.01)))
    return {
        "ticker": ticker,
        "metrics": {
            "initial_capital": 100000.0,
            "current_capital": round(equity[-1], 2),
            "total_return_pct": round((equity[-1] - 100000) / 1000, 2),
            "max_drawdown_pct": round(np.random.uniform(5, 12), 2),
            "sharpe_ratio": round(np.random.uniform(0.8, 2.5), 3),
            "sortino_ratio": round(np.random.uniform(1.0, 3.0), 3),
            "calmar_ratio": round(np.random.uniform(0.5, 2.0), 3),
            "win_rate_pct": round(np.random.uniform(52, 68), 1),
            "profit_factor": round(np.random.uniform(1.2, 2.5), 2),
            "total_trades": np.random.randint(20, 60),
        },
        "equity_curve": {"labels": [str(i) for i in range(len(equity))], "values": [round(v, 2) for v in equity]},
        "trades": [],
        "total_trades": 0,
        "demo": True,
    }


@app.get("/api/portfolio", response_model=PortfolioResponse)
async def get_portfolio() -> dict:
    """Get portfolio overview and risk metrics."""
    rm = state.risk_manager
    if rm is None:
        return {
            "initial_capital": 100000.0,
            "current_capital": 100000.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
            "trading_halted": False,
        }
    return rm.get_summary()


@app.get("/api/metrics")
async def get_metrics() -> dict:
    """Get model performance metrics."""
    return {
        "simulation_mode": state.simulation_mode,
        "loaded_tickers": list(state.ensembles.keys()),
        "available_tickers": state.tickers,
        "server_time": datetime.now().isoformat(),
        "connected_clients": len(state.connected_clients),
    }


class TrainRequest(BaseModel):
    ticker: str
    epochs: int = 10


@app.post("/api/train/{ticker}")
async def trigger_training(ticker: str, body: Optional[TrainRequest] = None) -> dict:
    """Trigger model training for a ticker."""
    ticker = ticker.upper()
    epochs = body.epochs if body else 10

    try:
        from src.data.fetcher import DataFetcher
        from src.data.preprocessor import FeaturePreprocessor
        from src.models.lstm_model import LSTMModel
        from src.models.transformer_model import TransformerModel
        from src.models.ensemble import EnsemblePredictor
        from src.training.trainer import ModelTrainer
        from src.strategy.signals import SignalGenerator

        fetcher = DataFetcher(tickers=[ticker], lookback_days=730)
        df = fetcher.fetch_historical(ticker)
        preprocessor = FeaturePreprocessor()
        X, y = preprocessor.fit_transform(df)

        trainer = ModelTrainer(epochs=epochs, batch_size=32)

        lstm = LSTMModel(n_features=X.shape[2])
        transformer = TransformerModel(n_features=X.shape[2])

        lstm_hist = trainer.train_model(lstm, X, y, f"{ticker}_lstm")
        trans_hist = trainer.train_model(transformer, X, y, f"{ticker}_transformer")

        lstm.save(f"models/saved/{ticker}/lstm_model.pt")
        transformer.save(f"models/saved/{ticker}/transformer_model.pt")
        preprocessor.save_scalers(f"models/saved/{ticker}/scaler.pkl")

        ensemble = EnsemblePredictor(lstm_model=lstm, transformer_model=transformer)
        state.ensembles[ticker] = ensemble
        state.preprocessors[ticker] = preprocessor
        state.signal_generators[ticker] = SignalGenerator()
        state.simulation_mode = False

        lstm_val = lstm_hist.get("val_loss", [0])
        trans_val = trans_hist.get("val_loss", [0])
        return {
            "ticker": ticker,
            "status": "trained",
            "lstm_val_loss": lstm_val[-1] if lstm_val else None,
            "transformer_val_loss": trans_val[-1] if trans_val else None,
            "epochs_completed": lstm_hist.get("epochs_completed", epochs),
        }
    except Exception as e:
        logger.error(f"Training failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
async def get_signal_history() -> list[dict]:
    """Get recent signal history (last 50)."""
    return state.signal_history[-50:]


# ── WebSocket ───────────────────────────────────────────────────


@app.websocket("/ws/stream/{ticker}")
async def websocket_endpoint(websocket: WebSocket, ticker: str) -> None:
    """WebSocket endpoint streaming real-time predictions every 5 seconds."""
    await websocket.accept()
    state.connected_clients.append(websocket)
    ticker = ticker.upper()
    logger.info(f"WebSocket connected: {ticker} (total: {len(state.connected_clients)})")

    try:
        while True:
            try:
                pred = state.get_or_create_prediction(ticker)

                # Also get signal
                SignalGenerator = _import_signal_generator()
                sig_gen = state.signal_generators.get(ticker, SignalGenerator())
                if ticker not in state.signal_generators:
                    state.signal_generators[ticker] = sig_gen

                indicators = {
                    "rsi": np.random.uniform(30, 70),
                    "ema_21": pred["current_price"] * 0.99,
                    "atr": pred["current_price"] * 0.015,
                }
                signal = sig_gen.generate(ticker, pred, pred["current_price"], indicators, pred["timestamp"])
                state.latest_signals[ticker] = signal

                payload = {
                    "type": "prediction",
                    "data": pred,
                    "signal": {
                        "type": signal.signal_type,
                        "strength": signal.strength,
                        "reason": signal.reason,
                        "stop_loss": signal.stop_loss,
                        "take_profit": signal.take_profit,
                    },
                }

                await websocket.send_json(payload)
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error for {ticker}: {e}")
                await asyncio.sleep(1)
                continue

            await asyncio.sleep(5)

    except WebSocketDisconnect:
        pass
    finally:
        if websocket in state.connected_clients:
            state.connected_clients.remove(websocket)
        logger.info(f"WebSocket disconnected: {ticker} (remaining: {len(state.connected_clients)})")


# ── Health ──────────────────────────────────────────────────────


@app.get("/api/health")
async def health_check() -> dict:
    return {
        "status": "ok",
        "simulation_mode": state.simulation_mode,
        "timestamp": datetime.now().isoformat(),
    }


# ── Background Scheduler ────────────────────────────────────────

_scheduler: Optional[BackgroundScheduler] = None


def _refresh_data_job() -> None:
    """Periodic data refresh job."""
    logger.debug("Background data refresh running...")


def start_scheduler() -> None:
    """Start background scheduler for periodic tasks."""
    global _scheduler
    if not SCHEDULER_AVAILABLE:
        logger.warning("APScheduler not available. Background tasks disabled.")
        return
    _scheduler = BackgroundScheduler()
    _scheduler.add_job(_refresh_data_job, "interval", seconds=30, id="data_refresh")
    _scheduler.start()
    logger.info("Background scheduler started")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean shutdown."""
    if _scheduler:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler shut down")


# ── Entry Point ─────────────────────────────────────────────────


def main() -> None:
    """Run the API server."""
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    try:
        start_scheduler()
    except Exception as e:
        logger.warning(f"Scheduler start failed: {e}")

    logger.info(f"Starting Trading Predictor API on {host}:{port}")
    logger.info(f"Mode: {'SIMULATION' if state.simulation_mode else 'LIVE'}")
    uvicorn.run("src.api.main:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
