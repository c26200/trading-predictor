#!/usr/bin/env bash
# TradeScope AI — Setup Script
# Installs all dependencies and prepares the project for first run.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  TRADESCOPE AI — Project Setup"
echo "=============================================="

# Python version check
echo ""
echo "[1/5] Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python $PYTHON_VERSION detected"

# Virtual environment
echo ""
echo "[2/5] Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  [OK] Virtual environment created"
else
    echo "  [OK] Virtual environment already exists"
fi

# Activate
source venv/bin/activate
echo "  [OK] Virtual environment activated"

# Install dependencies
echo ""
echo "[3/5] Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt
echo "  [OK] Dependencies installed"

# Create directories
echo ""
echo "[4/5] Creating project directories..."
mkdir -p models/saved logs models/saved/history
echo "  [OK] Directories created"

# Verify installation
echo ""
echo "[5/5] Verifying installation..."
python3 -c "
from src.data.fetcher import DataFetcher
from src.data.preprocessor import FeaturePreprocessor
print('  [OK] Data pipeline: OK')
"
python3 -c "
from src.api.main import app
print('  [OK] API server: OK')
" 2>/dev/null || echo "  [WARN] API imports need TensorFlow (optional for simulation mode)"

echo ""
echo "=============================================="
echo "  SETUP COMPLETE!"
echo "=============================================="
echo ""
echo "  Quick start:"
echo "    source venv/bin/activate"
echo "    python scripts/run_demo.py"
echo ""
echo "  Or manually:"
echo "    uvicorn src.api.main:app --reload"
echo "    open frontend/index.html"
echo ""
echo "  API Docs: http://localhost:8000/docs"
echo ""
