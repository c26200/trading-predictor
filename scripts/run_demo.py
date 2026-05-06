"""Launch the full trading predictor demo: start API server + open browser."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_PATH = PROJECT_ROOT / "frontend" / "index.html"


def main() -> None:
    """Start the backend server and open the dashboard in a browser."""
    print("=" * 60)
    print("  TRADESCOPE AI — Algorithmic Trading Dashboard")
    print("=" * 60)

    # Check dependencies
    print("\n[1/3] Checking dependencies...")
    try:
        import fastapi  # noqa
        import uvicorn  # noqa
        print("  [OK] FastAPI + Uvicorn available")
    except ImportError:
        print("  [ERROR] Install dependencies: pip install -r requirements.txt")
        sys.exit(1)

    # Start API server in background
    print("\n[2/3] Starting API server...")
    api_cmd = [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

    try:
        server_process = subprocess.Popen(
            api_cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"  [OK] API server starting (PID: {server_process.pid})")
    except Exception as e:
        print(f"  [ERROR] Failed to start server: {e}")
        sys.exit(1)

    # Wait for server to be ready
    print("  Waiting for server to be ready...")
    for i in range(15):
        time.sleep(1)
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:8000/api/health")
            print("  [OK] Server is ready!")
            break
        except Exception:
            if i == 14:
                print("  [WARN] Server may not be ready. Opening dashboard anyway...")

    # Open dashboard
    print("\n[3/3] Opening dashboard...")
    dashboard_url = f"file:///{str(FRONTEND_PATH).replace(chr(92), '/')}"
    print(f"  URL: {dashboard_url}")

    try:
        webbrowser.open(dashboard_url)
        print("  [OK] Dashboard opened in browser")
    except Exception:
        print(f"  [INFO] Open manually: {dashboard_url}")

    print("\n" + "=" * 60)
    print("  API Server : http://localhost:8000")
    print("  API Docs   : http://localhost:8000/docs")
    print("  Dashboard  : file:///" + str(FRONTEND_PATH).replace("\\", "/"))
    print("=" * 60)
    print("\nPress Ctrl+C to stop the server.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        server_process.terminate()
        server_process.wait(timeout=5)
        print("Server stopped. Goodbye!")


if __name__ == "__main__":
    main()
