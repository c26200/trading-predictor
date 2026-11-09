# TradeScope AI — Algorithmic Trading Prediction System

**1st place — International AI Festival 2025**

An end-to-end stock prediction system: a PyTorch ensemble (bidirectional LSTM with attention
plus a temporal transformer) served through FastAPI, with a browser dashboard that streams
predictions over WebSocket.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  FRONTEND                        │
│          Single-page Dashboard (HTML)            │
│    Lightweight Charts v4 + WebSocket Stream      │
└──────────────────┬──────────────────────────────┘
                   │ REST + WebSocket
┌──────────────────▼──────────────────────────────┐
│               FASTAPI SERVER                     │
│   /api/predict  /api/signals  /api/backtest      │
│   /api/portfolio  /api/metrics  /api/train       │
│           /ws/stream/{ticker}                    │
└──────┬────────────────┬──────────────┬───────────┘
       │                │              │
┌──────▼──────┐ ┌───────▼──────┐ ┌─────▼──────────┐
│  DATA PIPE  │ │  ML MODELS   │ │   STRATEGY      │
│  yfinance   │ │  LSTM +      │ │  Signals +      │
│  + Synth    │ │  Transformer │ │  Risk Mgmt +    │
│  Fallback   │ │  Ensemble    │ │  Backtester     │
└─────────────┘ └──────────────┘ └────────────────┘
```

## What it includes

- Bidirectional LSTM with attention and a temporal transformer, combined by a weighted
  ensemble that shifts weight toward whichever model performed better recently
- Signal engine with entry and exit rules, stop-loss, take-profit and trailing stops
- Risk management: Kelly-style position sizing, drawdown guards, and Sharpe, Sortino and
  Calmar ratios
- Walk-forward backtesting with commission and slippage simulation
- Simulation mode that generates synthetic prices, so the dashboard runs without network
  access
- WebSocket stream pushing a new prediction every five seconds

## Installation

```bash
git clone https://github.com/c26200/trading-predictor.git
cd trading-predictor

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`bash setup.sh` does the same thing in one step.

## Running

```bash
# Demo: starts the server and opens the dashboard
python scripts/run_demo.py

# Server only
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# then open frontend/index.html

# Train a model for one ticker
curl -X POST http://localhost:8000/api/train/AAPL -d '{"epochs": 10}'
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/predict/{ticker}` | Latest prediction with confidence |
| GET | `/api/signals/{ticker}` | Current entry and exit signals |
| GET | `/api/backtest/{ticker}` | Backtest results |
| GET | `/api/portfolio` | Portfolio overview |
| GET | `/api/metrics` | Model performance metrics |
| GET | `/api/history` | Recent signal history |
| POST | `/api/train/{ticker}` | Trigger model training |
| WS | `/ws/stream/{ticker}` | Real-time prediction stream |

## Supported tickers

Stocks: AAPL, TSLA, SPY, NVDA. Crypto: BTC-USD, ETH-USD.

## Project structure

```
trading-predictor/
├── src/
│   ├── data/           # Fetching and preprocessing
│   ├── models/         # LSTM, Transformer, ensemble
│   ├── strategy/       # Signals, risk, backtesting
│   ├── api/            # FastAPI server
│   └── training/       # Training pipeline
├── frontend/
│   └── index.html      # Single-file dashboard
├── scripts/            # Demo and training helpers
├── config/             # YAML configuration
├── models/saved/       # Trained model storage
└── requirements.txt
```

## Configuration

`config/config.yaml` controls the ticker list, data settings, model hyperparameters (LSTM
units, transformer heads), strategy thresholds (confidence, RSI, ATR multipliers) and the
API host and port.

## Stack

- **ML**: PyTorch 2.x
- **Data**: yfinance, pandas, numpy, ta
- **API**: FastAPI, Uvicorn, WebSockets
- **Frontend**: vanilla HTML/CSS/JS with Lightweight Charts v4
- **Scheduling**: APScheduler

## Notes

Trained weights are not committed — `models/saved/` ships empty, so you have to train a
ticker before predictions are meaningful. Backtest results depend on the date range and
data source; the backtester reports accuracy, Sharpe and maximum drawdown for each run
instead of fixed figures.

This is a research and portfolio project, not financial advice.
