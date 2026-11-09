"""End-to-end tests of the JSON API (Flask test client, real data, no mocks except the Gemini HTTP call)."""
import json

import pytest

from api.index import app
from src import ai


@pytest.fixture(scope="module")
def client():
    return app.test_client()


def test_health_and_meta(client):
    assert client.get("/api/health").json["status"] == "ok"
    meta = client.get("/api/meta").json
    assert len(meta["stores"]) == 45 and len(meta["models"]) == 8
    assert meta["dataset"]["rows"] > 400_000


def test_series_and_eda(client):
    s = client.get("/api/series?store=1&dept=1").json
    assert len(s["dates"]) == len(s["sales"]) == s["kpis"]["observations"]
    e = client.get("/api/eda?store=1&dept=1").json
    assert len(e["seasonal_profile"]) >= 50 and "holiday" in e


def test_evaluate_ranks_all_models(client):
    ev = client.get("/api/evaluate?store=1&dept=1").json
    assert len(ev["cv"]) == 8 and ev["winner"]["cv"]["rank"] == 1
    assert ev["winner"]["cv"]["wape"] < 8
    waps = [r["wape"] for r in ev["cv"]]
    assert waps == sorted(waps)
    assert ev["comparison"]["reduction_pct"] > 50
    json.dumps(ev)  # JSON-safe (no NaN)


def test_single_model_matches_full_evaluation(client):
    one = client.get("/api/evaluate/model?store=1&dept=1&model=naive").json
    full = client.get("/api/evaluate?store=1&dept=1").json
    row = next(r for r in full["cv"] if r["key"] == "naive")
    assert one["cv"]["wape"] == row["wape"]


def test_forecast_shape_and_bounds(client):
    f = client.get("/api/forecast?store=1&dept=1&horizon=6").json
    rows = f["forecast"]
    assert len(rows) == 6 and f["is_champion"]
    assert all(r["lower"] <= r["forecast"] <= r["upper"] for r in rows)
    widths = [r["upper"] - r["lower"] for r in rows]
    assert widths == sorted(widths)           # uncertainty grows with horizon
    assert client.get("/api/forecast?store=1&dept=1&horizon=99").status_code == 400


def test_future_calendar_uses_dataset_holidays(client):
    f = client.get("/api/forecast?store=1&dept=1&horizon=8").json
    thanks = [r for r in f["forecast"] if r["date"] == "2012-11-23"]
    assert thanks and thanks[0]["is_holiday"] == 1 and thanks[0]["holiday"] == "Thanksgiving"


def test_inventory_policy(client):
    r = client.post("/api/inventory", json={"forecast_weekly": 1000, "demand_std": 100, "lead_time_weeks": 2,
                                            "review_period_weeks": 1, "service_level_pct": 95,
                                            "current_on_hand": 500, "incoming_orders": 200}).json["policy"]
    assert r["safety_stock"] == pytest.approx(1.645 * 100 * 3 ** 0.5, rel=1e-3)
    assert r["order_quantity"] == pytest.approx(r["target_inventory"] - 700)
    assert client.post("/api/inventory", json={"forecast_weekly": "abc"}).status_code == 400


def test_simulation_steps_are_stateless_and_leak_free(client):
    base = {"store": 1, "dept": 1, "model": "improved_seasonal_regression", "horizon": 4}
    a = client.post("/api/simulate/step", json={**base, "origin": 100}).json
    b = client.post("/api/simulate/step", json={**base, "origin": 101, "state": a["state"]}).json
    assert b["kpis"]["steps"] == 2 and b["trained_on_weeks"] == a["trained_on_weeks"] + 1
    assert a["forecast"][0]["date"] == a["date"]       # forecast targets the revealed week
    assert client.post("/api/simulate/step", json={**base, "origin": 5}).status_code == 400


def test_validation_errors(client):
    assert client.get("/api/series?store=999&dept=1").status_code == 404
    assert client.get("/api/series?store=1&dept=99").status_code == 422
    assert client.get("/api/evaluate?store=1&dept=1&metric=bogus").status_code == 400


_AI_VARS = ["XAI_API_KEY", "GROK_API_KEY", "AI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "AI_PROVIDER", "AI_MODEL", "XAI_MODEL", "GEMINI_MODEL"]


def _clean_ai_env(monkeypatch):
    for v in _AI_VARS:
        monkeypatch.delenv(v, raising=False)
    ai._discover_model.cache_clear()


class _Resp:
    def __init__(self, payload): self.payload = payload
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def read(self): return json.dumps(self.payload).encode()


def test_ai_without_key_falls_back(client, monkeypatch):
    _clean_ai_env(monkeypatch)
    r = client.post("/api/ai/briefing", json={"store": 1, "dept": 1}).json
    assert r["source"] == "rules" and len(r["rule_based"]) == 5
    assert client.post("/api/ai/ask", json={"question": "hi"}).status_code == 503


def test_provider_detected_from_key(monkeypatch):
    _clean_ai_env(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "xai-abc123")          # a Grok key pasted under the wrong name
    assert ai.provider() == "xai"
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyExample")
    assert ai.provider() == "gemini"
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("GROK_API_KEY", "whatever")
    assert ai.provider() == "xai"


def test_grok_answer_with_model_discovery(client, monkeypatch):
    _clean_ai_env(monkeypatch)
    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    calls = []

    def fake_urlopen(req, timeout=0):
        calls.append((req.full_url, req.headers.get("Authorization")))
        if req.full_url.endswith("/models"):
            return _Resp({"data": [{"id": "grok-4"}, {"id": "grok-imagine-image"}, {"id": "grok-4-fast-non-reasoning"}]})
        body = json.loads(req.data)
        assert body["model"] == "grok-4-fast-non-reasoning" and "FACTS" in body["messages"][1]["content"]
        return _Resp({"choices": [{"message": {"content": "Stock up before week 51."}}]})

    monkeypatch.setattr(ai.urllib.request, "urlopen", fake_urlopen)
    r = client.post("/api/ai/ask", json={"store": 1, "dept": 1, "question": "When should I stock up?"}).json
    assert r["answer"].startswith("Stock up") and r["source"] == "xai"
    assert all(a == "Bearer xai-test-key" for _, a in calls) and "xai-test-key" not in json.dumps(r)
    assert client.get("/api/ai/status").json["provider"] == "xAI Grok"


def test_gemini_answer_with_explicit_model(client, monkeypatch):
    _clean_ai_env(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    monkeypatch.setenv("AI_MODEL", "gemini-test-model")
    seen = {}

    def fake_urlopen(req, timeout=0):
        seen["url"] = req.full_url; seen["key"] = req.headers.get("X-goog-api-key")
        return _Resp({"candidates": [{"content": {"parts": [{"text": "Hold extra stock in week 51."}]}}]})

    monkeypatch.setattr(ai.urllib.request, "urlopen", fake_urlopen)
    r = client.post("/api/ai/ask", json={"store": 1, "dept": 1, "question": "What next?"}).json
    assert r["answer"].startswith("Hold") and "gemini-test-model:generateContent" in seen["url"] and seen["key"] == "AIza-test"


def test_vercel_style_rewritten_path(client):
    """Vercel calls /api/index?__path=api/<route>; the real route and query must be restored."""
    r = client.get("/api/index?__path=api/series&store=2&dept=3")
    assert r.status_code == 200 and r.json["store"] == 2 and r.json["dept"] == 3
    assert client.get("/api/index?__path=api/health").json["status"] == "ok"
