# TradeScope AI — Algorithmic Trading Prediction System

**1st Place Winner — International AI Festival 2025**

A production-grade Stock Market Prediction System using Deep Learning (LSTM + Transformer ensemble) with a professional real-time dashboard.

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

## Features

- **Dual Deep Learning Models**: Bidirectional LSTM with Attention + Temporal Transformer
- **Weighted Ensemble**: Dynamic weight adjustment based on recent performance
- **Real-time Dashboard**: Bloomberg-style terminal with candlestick charts, indicators, and live signals
- **Trading Strategy Engine**: Entry/Exit signals with stop-loss, take-profit, trailing stops
- **Risk Management**: Kelly Criterion position sizing, max drawdown guards, Sharpe/Sortino/Calmar ratios
- **Walk-Forward Backtesting**: Commission and slippage simulation
- **Simulation Mode**: Works offline with synthetic data generation
- **WebSocket Streaming**: Real-time price predictions every 5 seconds

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd trading-predictor

# Run setup
bash setup.sh

# Or manually:
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

```bash
# Option 1: Demo script (starts server + opens browser)
python scripts/run_demo.py

# Option 2: Start server only
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# Then open frontend/index.html in your browser

# Option 3: Train models for a specific ticker
curl -X POST http://localhost:8000/api/train/AAPL -d '{"epochs": 10}'
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/predict/{ticker}` | Latest prediction with confidence |
| GET | `/api/signals/{ticker}` | Current Entry/Exit signals |
| GET | `/api/backtest/{ticker}` | Backtest results |
| GET | `/api/portfolio` | Portfolio overview |
| GET | `/api/metrics` | Model performance metrics |
| GET | `/api/history` | Recent signal history |
| POST | `/api/train/{ticker}` | Trigger model training |
| WS | `/ws/stream/{ticker}` | Real-time prediction stream |

## Supported Tickers

- **Stocks**: AAPL, TSLA, SPY, NVDA
- **Crypto**: BTC-USD, ETH-USD

## Performance Metrics (Example)

| Ticker | Win Rate | Sharpe | Max DD | Profit Factor |
|--------|----------|--------|--------|---------------|
| AAPL | 62.5% | 1.85 | 8.2% | 2.1 |
| TSLA | 58.3% | 1.42 | 12.1% | 1.7 |
| BTC-USD | 55.1% | 1.28 | 14.5% | 1.5 |
| SPY | 65.8% | 2.10 | 5.3% | 2.4 |
| NVDA | 60.2% | 1.75 | 9.8% | 1.9 |

*Note: Results shown are indicative. Live performance depends on market conditions and model retraining frequency.*

## Project Structure

```
trading-predictor/
├── src/
│   ├── data/           # Data fetching & preprocessing
│   ├── models/         # LSTM, Transformer, Ensemble
│   ├── strategy/       # Signals, Risk, Backtesting
│   ├── api/            # FastAPI server
│   └── training/       # Training pipeline
├── frontend/
│   └── index.html      # Single-file dashboard
├── scripts/            # Demo & training scripts
├── config/             # YAML configuration
├── models/saved/       # Trained model storage
├── logs/               # Application logs
└── requirements.txt
```

## Configuration

Edit `config/config.yaml` to customize:
- Ticker list and data settings
- Model hyperparameters (LSTM units, Transformer heads, etc.)
- Strategy thresholds (confidence, RSI, ATR multipliers)
- API host/port

## Technology Stack

- **ML**: TensorFlow 2.x (LSTM + Transformer)
- **Data**: yfinance, pandas, numpy, ta (technical analysis)
- **API**: FastAPI + Uvicorn + WebSockets
- **Frontend**: Vanilla HTML/CSS/JS + Lightweight Charts v4
- **Scheduling**: APScheduler

---

*Built for the International AI Festival 2025 — Algorithmic Trading Challenge*
