"""
Vercel entrypoint - Flask application exposing the forecasting engine as a JSON API.

Local development:   python scripts/dev_server.py
Vercel:              this file is detected automatically (variable ``app``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, request  # noqa: E402

from src import ai, service as S  # noqa: E402
from src.simulation import simulate_step, simulation_bounds  # noqa: E402

app = Flask(__name__, static_folder=None)
app.url_map.strict_slashes = False
app.json.sort_keys = False


# ----------------------------------------------------------------------------- helpers

def _int(name: str, default: int | None = None) -> int:
    raw = request.args.get(name)
    if raw is None:
        if default is None:
            raise S.ApiError(f"Missing query parameter '{name}'.")
        return default
    try:
        return int(raw)
    except ValueError:
        raise S.ApiError(f"'{name}' must be an integer.")


def _series_args() -> tuple[int, int]:
    return _int("store", 1), _int("dept", 1)


def _metric() -> str:
    return request.args.get("metric", "wape").lower()


def _body() -> dict:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise S.ApiError("Expected a JSON object body.")
    return data


def _client_id() -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return (fwd.split(",")[0].strip() or request.remote_addr or "anon")


@app.after_request
def _headers(resp):
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


@app.errorhandler(S.ApiError)
def _api_error(e: S.ApiError):
    return jsonify({"error": e.message}), e.status


@app.errorhandler(ai.AIError)
def _ai_error(e: ai.AIError):
    return jsonify({"error": e.message, "ai": ai.status()}), e.status


@app.errorhandler(404)
def _not_found(_):
    return jsonify({"error": "Not found."}), 404


@app.errorhandler(Exception)
def _unhandled(e):
    app.logger.exception("Unhandled error")
    return jsonify({"error": "Internal error while computing this request."}), 500


# ----------------------------------------------------------------------------- routes

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "ai": ai.status()})


@app.get("/api/meta")
def meta():
    return jsonify(S.get_meta())


@app.get("/api/series")
def series():
    store, dept = _series_args()
    return jsonify(S.get_series_payload(store, dept))


@app.get("/api/eda")
def eda():
    store, dept = _series_args()
    return jsonify(S.get_eda_payload(store, dept))


@app.get("/api/evaluate")
def evaluate():
    store, dept = _series_args()
    return jsonify(S.evaluate_all(store, dept, _metric()))


@app.get("/api/evaluate/model")
def evaluate_model():
    store, dept = _series_args()
    S.get_context(store, dept)
    key = request.args.get("model", "")
    return jsonify(S.evaluate_one_payload(store, dept, key))


@app.get("/api/forecast")
def forecast():
    store, dept = _series_args()
    return jsonify(S.forecast_payload(store, dept, request.args.get("model", "auto"),
                                      _int("horizon", 8), _metric()))


@app.post("/api/inventory")
def inventory():
    return jsonify(S.inventory_payload(_body()))


@app.get("/api/simulate/bounds")
def sim_bounds():
    store, dept = _series_args()
    return jsonify(S.jsonable(simulation_bounds(store, dept)))


@app.post("/api/simulate/step")
def sim_step():
    b = _body()
    try:
        store, dept = int(b.get("store", 1)), int(b.get("dept", 1))
        origin = int(b["origin"])
        kwargs = dict(
            horizon=int(b.get("horizon", 4)), lead_time=int(b.get("lead_time", 2)),
            review=int(b.get("review", 1)), service_level=float(b.get("service_level", 95.0)),
        )
    except (KeyError, TypeError, ValueError):
        raise S.ApiError("Invalid simulation parameters.")
    return jsonify(simulate_step(store, dept, str(b.get("model", "")), origin, state=b.get("state"), **kwargs))


@app.get("/api/ai/status")
def ai_status():
    return jsonify(ai.status())


@app.post("/api/ai/briefing")
def ai_briefing():
    b = _body()
    store, dept = int(b.get("store", 1)), int(b.get("dept", 1))
    S.get_context(store, dept)
    if ai.api_key():
        ai.check_rate_limit(_client_id())
    return jsonify(S.jsonable(ai.briefing(store, dept, str(b.get("metric", "wape")))))


@app.post("/api/ai/ask")
def ai_ask():
    b = _body()
    store, dept = int(b.get("store", 1)), int(b.get("dept", 1))
    S.get_context(store, dept)
    ai.check_rate_limit(_client_id())
    return jsonify(ai.ask(store, dept, str(b.get("question", "")), str(b.get("metric", "wape"))))


# ----------------------------------------------------------------------------- local dev only

if not os.environ.get("VERCEL"):
    from flask import send_from_directory

    PUBLIC = ROOT / "public"

    @app.get("/")
    def _index():
        return send_from_directory(PUBLIC, "index.html")

    @app.get("/<path:path>")
    def _static(path: str):
        target = PUBLIC / path
        if target.is_file():
            return send_from_directory(PUBLIC, path)
        if (PUBLIC / f"{path}.html").is_file():           # clean URLs, same as vercel.json
            return send_from_directory(PUBLIC, f"{path}.html")
        return jsonify({"error": "Not found."}), 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 3000)), debug=False)
