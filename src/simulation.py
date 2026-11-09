"""
simulation.py
Week-by-week "live replay" of the full production loop on historical data:

    for each simulated week t:
        1. receive orders that arrive this week
        2. RETRAIN the chosen model on weeks [0, t)  (no future data is visible)
        3. FORECAST the next H weeks recursively, with 95 % bounds
        4. apply the (R, S) replenishment policy -> place an order
        5. REVEAL the actual demand of week t, fulfil what stock allows, record lost sales
        6. update forecast-error statistics (used for the next safety stock)

The server is stateless: the inventory/accuracy state is returned to the caller and
sent back with the next step, so every step can be served by a different instance.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.forecasting import generate_future_forecast
from src.forecasting_models import get_all_models
from src.inventory import calculate_inventory_policy
from src.service import (
    ApiError,
    WARMUP_COLS,
    get_context,
    jsonable,
)

MIN_TRAIN_ROWS = 20          # post warm-up rows required to fit a model
MAX_HORIZON = 12


def simulation_bounds(store: int, dept: int) -> Dict[str, Any]:
    """Valid replay window for a series (first origin leaves >= MIN_TRAIN_ROWS usable rows)."""
    ctx = get_context(store, dept)
    n = len(ctx.series_df)
    first = 52 + MIN_TRAIN_ROWS
    return {"n": n, "min_origin": first, "max_origin": n - 1,
            "dates": ctx.series_df["Date"].dt.strftime("%Y-%m-%d").tolist(),
            "sales": ctx.series_df["Weekly_Sales"].astype(float).tolist()}


def _fresh_state() -> Dict[str, Any]:
    return {
        "on_hand": None, "pipeline": [], "errors": [], "first_origin": None,
        "totals": {"demand": 0.0, "fulfilled": 0.0, "unmet": 0.0, "stockout_weeks": 0,
                   "ordered": 0.0, "inventory_sum": 0.0, "steps": 0,
                   "abs_error": 0.0, "actual": 0.0},
    }


def _sanitize_state(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Accept a client-returned state but only trust well-formed numbers."""
    st = _fresh_state()
    if not raw:
        return st
    try:
        if raw.get("on_hand") is not None:
            st["on_hand"] = max(0.0, float(raw["on_hand"]))
        st["pipeline"] = [[int(a), max(0.0, float(q))] for a, q in raw.get("pipeline", [])][:64]
        st["errors"] = [float(e) for e in raw.get("errors", [])][-104:]
        if raw.get("first_origin") is not None:
            st["first_origin"] = int(raw["first_origin"])
        for k in st["totals"]:
            if k in raw.get("totals", {}):
                st["totals"][k] = float(raw["totals"][k])
    except (TypeError, ValueError):
        raise ApiError("Malformed simulation state.")
    return st


def simulate_step(store: int, dept: int, model_key: str, origin: int, horizon: int = 4,
                  lead_time: int = 2, review: int = 1, service_level: float = 95.0,
                  state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    t_start = time.perf_counter()
    ctx = get_context(store, dept)
    n = len(ctx.series_df)
    first = 52 + MIN_TRAIN_ROWS
    if not (first <= origin < n):
        raise ApiError(f"origin must be between {first} and {n - 1} for this series.")
    if model_key not in get_all_models():
        raise ApiError(f"Unknown model '{model_key}'.", 404)
    lead_time = int(min(max(lead_time, 1), 8))
    review = int(min(max(review, 1), 4))
    horizon = int(min(max(horizon, lead_time + review, 1), MAX_HORIZON))
    service_level = float(min(max(service_level, 50.0), 99.9))

    st = _sanitize_state(state)
    if st["first_origin"] is None:
        st["first_origin"] = origin

    series = ctx.series_df
    dates = series["Date"]
    sales = series["Weekly_Sales"].astype(float).values
    this_date = dates.iloc[origin]

    # ---- 1. receipts --------------------------------------------------------------------
    receipts = sum(q for a, q in st["pipeline"] if a <= origin)
    st["pipeline"] = [[a, q] for a, q in st["pipeline"] if a > origin]

    # ---- 2. retrain on the past only ---------------------------------------------------
    t_fit = time.perf_counter()
    past_feat = ctx.feat_df.iloc[:origin]
    train = past_feat.dropna(subset=[c for c in WARMUP_COLS if c in past_feat.columns]).reset_index(drop=True)
    if len(train) < MIN_TRAIN_ROWS:
        raise ApiError("Not enough history to retrain at this origin.", 422)
    model = get_all_models()[model_key]
    if model.model_type in ("ml", "baseline") and getattr(model, "feature_names", None):
        model.fit(y=train["Weekly_Sales"], X=train)
    else:
        model.fit(y=train["Weekly_Sales"])
    fit_ms = int(round((time.perf_counter() - t_fit) * 1000))

    # ---- 3. forecast --------------------------------------------------------------------
    recent = train["Weekly_Sales"].tail(52)
    errs = st["errors"]
    if len(errs) >= 4:
        sigma = float(math.sqrt(np.mean(np.square(errs))))      # RMSE of own past forecasts
        sigma_src = "observed forecast RMSE"
    else:
        sigma = float(recent.std() * 0.15)
        sigma_src = "warm-up estimate (15% of 52-wk std)"
    sigma = max(sigma, 1.0)
    last_hist_date = pd.to_datetime(dates.iloc[origin - 1])
    cal = ctx.calendar[ctx.calendar["Date"] > last_hist_date]
    fut = generate_future_forecast(model, series.iloc[:origin], horizon=horizon,
                                   residual_std=sigma, z_score=1.96, future_features=cal)
    f_vals = fut["Forecast ($)"].astype(float).tolist()

    # ---- 4. replenishment policy ----------------------------------------------------------
    protection = lead_time + review
    weekly_f = float(np.mean(f_vals[:protection]))
    on_order = float(sum(q for _, q in st["pipeline"]))
    policy_kw = dict(forecast_weekly=weekly_f, demand_std=sigma, lead_time_weeks=lead_time,
                     review_period_weeks=review, service_level_pct=service_level,
                     backorders=0.0, is_monetary_scenario=True)

    if st["on_hand"] is None:                       # start the replay fully stocked at S
        init = calculate_inventory_policy(current_on_hand=0.0, incoming_orders=0.0, **policy_kw)
        st["on_hand"] = float(init["target_inventory"])
        initial_stock = st["on_hand"]
    else:
        initial_stock = None
    on_hand_start = st["on_hand"] + receipts

    policy = calculate_inventory_policy(current_on_hand=on_hand_start, incoming_orders=on_order, **policy_kw)
    is_review_week = ((origin - st["first_origin"]) % review) == 0
    order_qty = float(policy["order_quantity"]) if is_review_week else 0.0
    if order_qty > 0:
        st["pipeline"].append([origin + lead_time, order_qty])

    # ---- 5. demand is revealed ----------------------------------------------------------
    demand = float(sales[origin])
    available = on_hand_start
    fulfilled = min(max(demand, 0.0), available)
    unmet = max(demand, 0.0) - fulfilled
    on_hand_end = available - fulfilled
    stockout = unmet > 1e-9

    # ---- 6. accuracy bookkeeping -----------------------------------------------------------
    error = demand - f_vals[0]
    st["errors"].append(error)
    tot = st["totals"]
    tot["demand"] += max(demand, 0.0)
    tot["fulfilled"] += fulfilled
    tot["unmet"] += unmet
    tot["stockout_weeks"] += int(stockout)
    tot["ordered"] += order_qty
    tot["inventory_sum"] += on_hand_end
    tot["steps"] += 1
    tot["abs_error"] += abs(error)
    tot["actual"] += abs(demand)
    st["on_hand"] = on_hand_end

    arrive_idx = origin + lead_time
    arrive_date = (dates.iloc[arrive_idx].strftime("%Y-%m-%d") if arrive_idx < n
                   else (pd.Timestamp(this_date) + pd.Timedelta(weeks=lead_time)).strftime("%Y-%m-%d"))

    steps = max(tot["steps"], 1)
    kpis = {
        "running_wape": (tot["abs_error"] / tot["actual"] * 100.0) if tot["actual"] > 0 else None,
        "fill_rate": (tot["fulfilled"] / tot["demand"] * 100.0) if tot["demand"] > 0 else None,
        "stockout_weeks": int(tot["stockout_weeks"]),
        "avg_inventory": tot["inventory_sum"] / steps,
        "total_ordered": tot["ordered"],
        "lost_sales": tot["unmet"],
        "steps": int(tot["steps"]),
    }

    return jsonable({
        "origin": origin, "date": this_date, "model": {"key": model_key, "name": model.name},
        "trained_on_weeks": int(len(train)), "fit_ms": fit_ms,
        "elapsed_ms": int(round((time.perf_counter() - t_start) * 1000)),
        "forecast": [{"date": r["Date"], "forecast": r["Forecast ($)"], "lower": r["Lower_Bound ($)"],
                      "upper": r["Upper_Bound ($)"], "holiday": r["Holiday"]} for _, r in fut.iterrows()],
        "actual": demand, "error": error,
        "pct_error": (abs(error) / demand * 100.0) if demand > 0 else None,
        "sigma": sigma, "sigma_source": sigma_src,
        "inventory": {
            "initial_stock": initial_stock,
            "receipts": receipts, "on_hand_start": on_hand_start, "on_order": on_order,
            "safety_stock": policy["safety_stock"], "reorder_point": policy["reorder_point"],
            "target_inventory": policy["target_inventory"],
            "inventory_position": policy["current_inventory_position"],
            "order_qty": order_qty, "order_arrives": arrive_date, "is_review_week": is_review_week,
            "fulfilled": fulfilled, "unmet": unmet, "stockout": stockout, "on_hand_end": on_hand_end,
        },
        "kpis": kpis,
        "state": st,
        "done": origin >= n - 1,
    })
