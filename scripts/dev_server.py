"""Run the full app (static pages + API) locally: python scripts/dev_server.py"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.index import app  # noqa: E402

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    print(f"\n  Retail Demand Forecasting  ->  http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
