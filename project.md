You are an expert quantitative developer and ML engineer. Build a complete, production-grade Stock Market Prediction System with Deep Learning for algorithmic trading. This must be a WINNING festival project demonstrating cutting-edge innovation.

## PROJECT OVERVIEW
Build a full-stack algorithmic trading prediction system using LSTM and Transformer architectures in Python, with a professional real-time dashboard. Everything must work end-to-end with no placeholder code.

---

## ARCHITECTURE

### 1. BACKEND — Python ML Engine (`src/`)

**Data Pipeline (`src/data/`)**
- `fetcher.py`: Fetch OHLCV data using `yfinance` for multiple tickers (AAPL, TSLA, BTC-USD, ETH-USD, SPY, NVDA). Support historical (2 years) and simulated real-time streaming (WebSocket or polling every 5s).
- `preprocessor.py`: Feature engineering pipeline:
  - Technical indicators via `ta` library: RSI, MACD, Bollinger Bands, ATR, EMA(9,21,50), Stochastic Oscillator, OBV, VWAP
  - Lag features (t-1 to t-10)
  - Returns, log-returns, volatility windows (5, 10, 20)
  - MinMaxScaler / RobustScaler with saved scaler state
  - Sequence windowing (look_back=60 days)

**Models (`src/models/`)**
- `lstm_model.py`: Stacked Bidirectional LSTM
  - 3 layers: 256 → 128 → 64 units, bidirectional
  - Dropout(0.3) between layers
  - Attention mechanism layer (custom Keras layer)
  - Output: next N-step price prediction + direction classification
  - Residual connections
  
- `transformer_model.py`: Temporal Transformer
  - Multi-head self-attention (8 heads, d_model=128)
  - Positional encoding (sinusoidal)
  - 4 encoder blocks with Feed-Forward sublayers
  - Layer normalization + dropout
  - Dual output head: regression (price) + classification (up/down/neutral)

- `ensemble.py`: Weighted ensemble combining LSTM + Transformer
  - Dynamic weight adjustment based on recent validation performance
  - Confidence scoring per prediction (0–100%)
  - Output: predicted price, direction, confidence score

**Trading Strategy Engine (`src/strategy/`)**
- `signals.py`: Entry/Exit signal generator
  - LONG signal: confidence > 75%, direction = UP, RSI < 65, price above EMA21
  - SHORT signal: confidence > 75%, direction = DOWN, RSI > 35, price below EMA21
  - EXIT signal: stop-loss (2% ATR-based), take-profit (4% ATR-based), trailing stop
  - Signal strength scoring (1–5 stars)

- `risk_manager.py`: Portfolio risk management
  - Position sizing: Kelly Criterion + fixed fractional (2% max risk per trade)
  - Max drawdown guard (halt trading if DD > 15%)
  - Correlation-based portfolio exposure limits
  - Sharpe, Sortino, Calmar ratio calculation

- `backtester.py`: Full backtesting engine
  - Walk-forward validation (expanding window)
  - Commission model (0.1% per trade)
  - Slippage simulation
  - Metrics: Total Return, Win Rate, Profit Factor, Max Drawdown, Sharpe Ratio, Sortino Ratio
  - Equity curve generation

**API Server (`src/api/`)**
- `main.py`: FastAPI application
  - WebSocket endpoint `/ws/stream/{ticker}` — streams real-time predictions every 5 seconds
  - REST endpoints:
    - `GET /api/predict/{ticker}` — latest prediction with confidence
    - `GET /api/signals/{ticker}` — current Entry/Exit signals
    - `GET /api/backtest/{ticker}` — backtest results
    - `GET /api/portfolio` — portfolio overview
    - `GET /api/metrics` — model performance metrics
    - `POST /api/train/{ticker}` — trigger model training
  - CORS enabled for frontend
  - Background task scheduler (APScheduler) for periodic retraining and data refresh

**Training Pipeline (`src/training/`)**
- `trainer.py`:
  - Early stopping, ReduceLROnPlateau callbacks
  - ModelCheckpoint saving best weights
  - Hyperparameter configs via `config.yaml`
  - Training history logging to JSON
  - GPU support auto-detection (TensorFlow/PyTorch)

---

### 2. FRONTEND — Professional Trading Dashboard (`frontend/`)

Single HTML file (`frontend/index.html`) — fully self-contained, no build step required.

**Visual Design**
- Dark terminal/Bloomberg aesthetic: background `#0a0e1a`, accent `#00ff88` (green), `#ff3366` (red), `#00b4ff` (blue)
- Monospace + futuristic font combination (JetBrains Mono + Orbitron from Google Fonts)
- Animated scanline overlay on header
- Glowing borders on cards, neon drop shadows
- Smooth CSS transitions on all data updates

**Layout (CSS Grid)**
┌─────────────────────────────────────────────────────┐
│  HEADER: Logo + Live Clock + Connection Status LED  │
├──────────────┬──────────────────────┬───────────────┤
│ TICKER       │  MAIN PRICE CHART    │ PORTFOLIO     │
│ SELECTOR     │  (TradingView-like)  │ SUMMARY       │
│ Panel        │  + Prediction overlay│               │
├──────────────┼──────────────────────┼───────────────┤
│ SIGNAL       │  INDICATORS PANEL    │ BACKTEST      │
│ PANEL        │  (RSI, MACD, BB)    │ RESULTS       │
│ Entry/Exit   │                      │ Equity Curve  │
├──────────────┴──────────────────────┴───────────────┤
│  LIVE TRADE LOG (scrolling table, last 50 signals)  │
└─────────────────────────────────────────────────────┘

**Charts (Lightweight Charts v4 from CDN)**
- Main candlestick chart with volume histogram
- Prediction line overlay (dashed, colored by confidence)
- Entry markers (▲ green triangles) and Exit markers (▼ red triangles)
- Bollinger Bands overlay
- RSI oscillator sub-chart
- MACD sub-chart
- Equity curve line chart (backtest panel)

**Real-time Features**
- WebSocket connection to backend with auto-reconnect
- Animated price ticker with color flash on change (green UP, red DOWN)
- Live signal alerts with sound toggle (Web Audio API beep on new signal)
- Confidence meter: animated arc/gauge (SVG)
- Model performance live counter: Win Rate, Sharpe, Total Return
- Loading skeleton animations while fetching

**Simulation Mode (fallback if backend offline)**
- Generate realistic synthetic OHLCV data (geometric Brownian motion)
- Simulate predictions with noise
- Show demo signals — so the dashboard is NEVER empty
- Clear "SIMULATION MODE" badge shown

---

### 3. CONFIGURATION (`config/`)
- `config.yaml`: All hyperparameters, ticker list, model paths, API settings
- `requirements.txt`: All dependencies pinned
- `.env.example`: API keys template

---

### 4. SCRIPTS
- `scripts/train_all.py`: Train models for all tickers sequentially
- `scripts/run_demo.py`: Start backend + open browser automatically
- `setup.sh`: Install all dependencies

---

### 5. DOCUMENTATION (`README.md`)
Professional README with:
- Project architecture diagram (ASCII)
- Installation steps
- How to train models
- How to run
- Performance metrics table (example results)
- Festival context: "1st Place Winner — International AI Festival 2025"
- Screenshots section placeholder

---

## TECHNICAL REQUIREMENTS

- Python 3.10+
- TensorFlow 2.x OR PyTorch 2.x (use TensorFlow as primary, add PyTorch Transformer option)
- FastAPI + uvicorn
- yfinance, ta, pandas, numpy, scikit-learn, scipy
- APScheduler for background tasks
- Lightweight Charts v4 (CDN) for frontend charts
- All models save/load from `models/saved/` directory
- Logging: Python logging module, structured JSON logs to `logs/`
- Error handling: try/except everywhere, graceful degradation
- Type hints throughout all Python code

---

## QUALITY STANDARDS

1. **Zero placeholder code** — every function must be fully implemented
2. **Error handling** — all API calls, file I/O, and model inference wrapped in try/except
3. **Simulation fallback** — if yfinance fails, use synthetic data so demo never crashes
4. **Models pre-trained on startup** — if no saved model exists, auto-train on first run (lightweight config: 10 epochs, 1 ticker)
5. **Professional comments** — docstrings on every class and function
6. **Performance** — predictions must respond in < 500ms

---

## FILE STRUCTURE
trading-predictor/
├── src/
│   ├── data/
│   │   ├── init.py
│   │   ├── fetcher.py
│   │   └── preprocessor.py
│   ├── models/
│   │   ├── init.py
│   │   ├── lstm_model.py
│   │   ├── transformer_model.py
│   │   └── ensemble.py
│   ├── strategy/
│   │   ├── init.py
│   │   ├── signals.py
│   │   ├── risk_manager.py
│   │   └── backtester.py
│   ├── api/
│   │   ├── init.py
│   │   └── main.py
│   └── training/
│       ├── init.py
│       └── trainer.py
├── frontend/
│   └── index.html
├── scripts/
│   ├── train_all.py
│   └── run_demo.py
├── config/
│   └── config.yaml
├── models/
│   └── saved/  (gitkeep)
├── logs/       (gitkeep)
├── requirements.txt
├── setup.sh
└── README.md

---

## EXECUTION ORDER

Build files in this exact order:
1. `config/config.yaml`
2. `requirements.txt`
3. `src/data/fetcher.py`
4. `src/data/preprocessor.py`
5. `src/models/lstm_model.py`
6. `src/models/transformer_model.py`
7. `src/models/ensemble.py`
8. `src/strategy/signals.py`
9. `src/strategy/risk_manager.py`
10. `src/strategy/backtester.py`
11. `src/training/trainer.py`
12. `src/api/main.py`
13. `frontend/index.html`
14. `scripts/run_demo.py`
15. `setup.sh`
16. `README.md`

After creating all files, run: `pip install -r requirements.txt` and verify no import errors exist across all modules by running `python -c "from src.api.main import app; print('OK')"`.

Start building now. Do not ask clarifying questions — implement everything completely.