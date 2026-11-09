"""
service.py
Application service layer: orchestrates the forecasting engine and returns
JSON-safe payloads for the web API. Everything is computed live from the raw
CSV files; results are memoised per warm instance so repeat calls are instant.
"""

from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data_loader import (
    PROJECT_ROOT,
    load_raw_datasets,
    prepare_merged_dataset,
    validate_dataset,
    get_series,
)
from src.preprocessing import preprocess_series, MARKDOWN_INTRO_DATE
from src.feature_engineering import build_feature_dataframe, FEATURE_SETS
from src.forecasting_models import get_all_models
from src.evaluation import (
    evaluate_holdout,
    evaluate_rolling_origin,
    build_ranked_leaderboard,
    analyze_residuals_and_holidays,
    evaluate_holdout_shocks_all_models,
)
from src.forecasting import generate_future_forecast, KNOWN_HOLIDAY_WEEKS
from src.inventory import calculate_inventory_policy, SERVICE_LEVEL_Z

warnings.filterwarnings("ignore")

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))

#: A series needs 52 warm-up weeks (Lag_52) plus enough rows to train and validate.
MIN_MODEL_OBS = 100
CV_HORIZON = 4
CV_SPLITS = 4

METRICS = {
    "wape": "WAPE (%)",
    "rmse": "RMSE ($)",
    "mae": "MAE ($)",
    "mape": "MAPE (%)",
}

MODEL_FAMILY_LABEL = {"baseline": "Baseline", "statistical": "Statistical", "ml": "Machine Learning"}

WARMUP_COLS = ["Lag_1", "Lag_2", "Lag_3", "Lag_4", "Lag_52", "Rolling_Mean_13"]


class ApiError(Exception):
    """Raised for client-correctable problems; mapped to a JSON error response."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def jsonable(obj: Any) -> Any:
    """Recursively convert numpy / pandas objects (and NaN/inf) into JSON-safe values."""
    if obj is None:
        return None
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, np.ndarray):
        return [jsonable(x) for x in obj.tolist()]
    if isinstance(obj, pd.Series):
        return [jsonable(x) for x in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return [jsonable(r) for r in obj.to_dict(orient="records")]
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    return obj


def _dates(series: pd.Series) -> List[str]:
    return pd.to_datetime(series).dt.strftime("%Y-%m-%d").tolist()


def _ms(start: float) -> int:
    return int(round((time.perf_counter() - start) * 1000))


# ---------------------------------------------------------------------------
# Dataset + per-series context (cached)
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    merged: pd.DataFrame
    report: Dict[str, Any]
    stores: pd.DataFrame
    features: pd.DataFrame
    test: Optional[pd.DataFrame]
    obs_by_series: Dict[int, Dict[int, int]]


@lru_cache(maxsize=1)
def get_dataset() -> Dataset:
    train, features, stores, test = load_raw_datasets(DATA_DIR)
    merged = prepare_merged_dataset(train, features, stores)
    report = validate_dataset(merged)

    feats = features.copy()
    feats["Date"] = pd.to_datetime(feats["Date"])
    feats["IsHoliday"] = feats["IsHoliday"].astype(bool)

    counts = merged.groupby(["Store", "Dept"]).size()
    obs: Dict[int, Dict[int, int]] = {}
    for (s, d), n in counts.items():
        obs.setdefault(int(s), {})[int(d)] = int(n)

    return Dataset(merged=merged, report=report, stores=stores, features=feats, test=test, obs_by_series=obs)


@dataclass
class SeriesContext:
    store: int
    dept: int
    series_df: pd.DataFrame      # raw merged rows for the series
    diagnostics: Dict[str, Any]
    clean_df: pd.DataFrame       # preprocessed
    feat_df: pd.DataFrame        # engineered features (all rows, warm-up rows contain NaN)
    clean_history: pd.DataFrame  # feature rows after warm-up
    calendar: pd.DataFrame       # per-date known covariates for the store (history + future)


def validate_series_ids(store: int, dept: int) -> None:
    ds = get_dataset()
    if store not in ds.obs_by_series:
        raise ApiError(f"Store {store} does not exist.", 404)
    if dept not in ds.obs_by_series[store]:
        raise ApiError(f"Department {dept} is not sold in store {store}.", 404)
    n = ds.obs_by_series[store][dept]
    if n < MIN_MODEL_OBS:
        raise ApiError(
            f"Store {store} / Dept {dept} has only {n} weekly observations; "
            f"at least {MIN_MODEL_OBS} are needed for 52-week seasonality and validation.",
            422,
        )


@lru_cache(maxsize=96)
def get_context(store: int, dept: int) -> SeriesContext:
    validate_series_ids(store, dept)
    ds = get_dataset()
    series_df, diag = get_series(ds.merged, store_id=store, dept_id=dept, min_observations=MIN_MODEL_OBS)
    clean = preprocess_series(series_df)
    feat = build_feature_dataframe(clean)
    hist = feat.dropna(subset=[c for c in WARMUP_COLS if c in feat.columns]).reset_index(drop=True)
    cal = ds.features[ds.features["Store"] == store].sort_values("Date").reset_index(drop=True)
    return SeriesContext(store, dept, series_df, diag, clean, feat, hist, cal)


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------

def model_catalog() -> List[Dict[str, Any]]:
    out = []
    for key, m in get_all_models().items():
        out.append({
            "key": key,
            "name": m.name,
            "type": m.model_type,
            "family": MODEL_FAMILY_LABEL.get(m.model_type, m.model_type),
            "description": m.description,
            "features": list(getattr(m, "feature_names", []) or []),
        })
    return out


def get_meta() -> Dict[str, Any]:
    ds = get_dataset()
    store_rows = []
    for _, r in ds.stores.sort_values("Store").iterrows():
        sid = int(r["Store"])
        depts = [[d, n] for d, n in sorted(ds.obs_by_series.get(sid, {}).items())]
        store_rows.append({
            "id": sid,
            "type": str(r["Type"]),
            "size": int(r["Size"]),
            "depts": depts,
        })
    rep = ds.report
    test_rows = int(len(ds.test)) if ds.test is not None else 0
    return jsonable({
        "stores": store_rows,
        "min_model_obs": MIN_MODEL_OBS,
        "metrics": METRICS,
        "models": model_catalog(),
        "service_levels": sorted(SERVICE_LEVEL_Z.keys()),
        "holiday_weeks": {str(k): v for k, v in KNOWN_HOLIDAY_WEEKS.items()},
        "markdown_intro_date": MARKDOWN_INTRO_DATE,
        "dataset": {
            "rows": rep["total_rows"],
            "stores": rep["unique_stores"],
            "departments": rep["unique_depts"],
            "date_min": rep["date_min"],
            "date_max": rep["date_max"],
            "duplicate_keys": rep["duplicate_keys"],
            "negative_sales_count": rep["negative_sales_count"],
            "negative_sales_pct": round(rep["negative_sales_pct"], 3),
            "missing_by_column": rep["missing_by_column"],
            "future_calendar_rows": test_rows,
        },
        "defaults": {"store": 1, "dept": 1, "metric": "wape"},
    })


# ---------------------------------------------------------------------------
# Series + KPIs
# ---------------------------------------------------------------------------

def get_series_payload(store: int, dept: int) -> Dict[str, Any]:
    t0 = time.perf_counter()
    ctx = get_context(store, dept)
    s = ctx.series_df
    sales = s["Weekly_Sales"].astype(float)
    rolling4 = sales.rolling(window=4, min_periods=1).mean()

    x = np.arange(len(sales))
    slope, intercept = np.polyfit(x, sales.values, 1)
    trend = slope * x + intercept

    i_max, i_min = int(sales.idxmax()), int(sales.idxmin())
    store_row = get_dataset().stores
    store_row = store_row[store_row["Store"] == store].iloc[0]

    diag = ctx.diagnostics
    return jsonable({
        "store": store,
        "dept": dept,
        "store_type": str(store_row["Type"]),
        "store_size": int(store_row["Size"]),
        "dates": _dates(s["Date"]),
        "sales": sales,
        "rolling4": rolling4,
        "trend": trend,
        "trend_slope_per_week": float(slope),
        "is_holiday": s["IsHoliday"].astype(int),
        "kpis": {
            "observations": int(len(s)),
            "total_revenue": float(sales.sum()),
            "avg_weekly": float(sales.mean()),
            "std_weekly": float(sales.std()),
            "latest_sales": float(sales.iloc[-1]),
            "latest_date": s["Date"].iloc[-1],
            "peak_sales": float(sales.iloc[i_max]),
            "peak_date": s["Date"].iloc[i_max],
            "low_sales": float(sales.iloc[i_min]),
            "low_date": s["Date"].iloc[i_min],
            "holiday_weeks": int((s["IsHoliday"] == True).sum()),  # noqa: E712
            "negative_weeks": int((sales < 0).sum()),
        },
        "diagnostics": {
            "start_date": diag["start_date"],
            "end_date": diag["end_date"],
            "gap_count": diag["gap_count"],
            "has_gaps": diag["has_gaps"],
            "warning": diag["warning"],
            "warmup_weeks": int(len(ctx.feat_df) - len(ctx.clean_history)),
            "model_rows": int(len(ctx.clean_history)),
        },
        "elapsed_ms": _ms(t0),
    })


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------

def get_eda_payload(store: int, dept: int) -> Dict[str, Any]:
    t0 = time.perf_counter()
    ctx = get_context(store, dept)
    df = ctx.clean_df.copy()
    df["Week_of_Year"] = df["Date"].dt.isocalendar().week.astype(int)

    # --- annual seasonal profile
    prof = df.groupby("Week_of_Year")["Weekly_Sales"].agg(["mean", "std", "count"]).reset_index()
    prof["std"] = prof["std"].fillna(0.0)
    top = prof.sort_values("mean", ascending=False).head(5)
    low = prof.sort_values("mean", ascending=True).head(5)

    # --- holiday vs normal weeks
    df["Period_Type"] = np.where(df["IsHoliday"] == 1, "Holiday Week", "Non-Holiday Week")
    df["Event_Name"] = [
        KNOWN_HOLIDAY_WEEKS.get(int(w), "Other Holiday Week") if h == 1 else "Regular Non-Holiday"
        for w, h in zip(df["Week_of_Year"], df["IsHoliday"])
    ]
    total_n = len(df)
    summary = df.groupby("Period_Type")["Weekly_Sales"].agg(["count", "mean", "median", "std"]).reset_index()
    non_mask = summary["Period_Type"] == "Non-Holiday Week"
    hol_mask = summary["Period_Type"] == "Holiday Week"
    non_mean = float(summary.loc[non_mask, "mean"].iloc[0]) if non_mask.any() else 1.0
    hol_mean = float(summary.loc[hol_mask, "mean"].iloc[0]) if hol_mask.any() else non_mean
    uplift = (hol_mean - non_mean) / non_mean * 100.0 if non_mean else 0.0
    n_hol = int(summary.loc[hol_mask, "count"].iloc[0]) if hol_mask.any() else 0

    period_stats = []
    for _, r in summary.iterrows():
        period_stats.append({
            "period": r["Period_Type"], "n": int(r["count"]), "mean": float(r["mean"]),
            "median": float(r["median"]), "std": 0.0 if pd.isna(r["std"]) else float(r["std"]),
            "share_pct": round(float(r["count"]) / total_n * 100.0, 1),
        })

    ev = df.groupby("Event_Name")["Weekly_Sales"].agg(["count", "mean", "median", "std"]).reset_index()
    events = []
    for _, r in ev.iterrows():
        events.append({
            "event": r["Event_Name"], "n": int(r["count"]), "mean": float(r["mean"]),
            "median": float(r["median"]), "std": 0.0 if pd.isna(r["std"]) else float(r["std"]),
            "uplift_pct": round((float(r["mean"]) - non_mean) / non_mean * 100.0, 1) if non_mean else 0.0,
        })

    box = {
        "holiday": df.loc[df["IsHoliday"] == 1, "Weekly_Sales"].astype(float).tolist(),
        "non_holiday": df.loc[df["IsHoliday"] != 1, "Weekly_Sales"].astype(float).tolist(),
    }

    # --- promotions
    md_cols = [c for c in ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"] if c in df.columns]
    promo_df = df[df["MarkDown_Available"] == 1]
    markdown = {
        "dates": _dates(promo_df["Date"]),
        "series": {c: promo_df[c].astype(float).tolist() for c in md_cols},
    }
    # Same definition as the original notebook: any week with a positive total markdown is a promo week.
    # NOTE: markdowns were not recorded before Nov-2011, so this comparison is confounded by time.
    promo_cmp = {"available_weeks": int(len(promo_df)), "total_weeks": int(len(df))}
    active = df[df["Total_MarkDown"] > 0]["Weekly_Sales"]
    inactive = df[df["Total_MarkDown"] <= 0]["Weekly_Sales"]
    promo_cmp.update({
        "with_promo_mean": float(active.mean()) if len(active) else None,
        "with_promo_n": int(len(active)),
        "without_promo_mean": float(inactive.mean()) if len(inactive) else None,
        "without_promo_n": int(len(inactive)),
    })
    top_promos = (
        ctx.calendar.assign(Total_MarkDown=ctx.calendar[md_cols].fillna(0).sum(axis=1))
        .query("Total_MarkDown > 0")
        .nlargest(10, "Total_MarkDown")[["Date", "Total_MarkDown", "IsHoliday"]]
    )
    top_promos["IsHoliday"] = top_promos["IsHoliday"].astype(int)

    # --- macro drivers (Pearson correlation with weekly sales)
    drivers = []
    for col, label in [("Temperature", "Temperature"), ("Fuel_Price", "Fuel price"),
                       ("CPI", "CPI"), ("Unemployment", "Unemployment"),
                       ("Total_MarkDown", "Total markdown"), ("IsHoliday", "Holiday flag")]:
        if col in df.columns and df[col].std() > 0:
            corr = float(np.corrcoef(df[col].astype(float), df["Weekly_Sales"].astype(float))[0, 1])
            drivers.append({"feature": label, "correlation": corr})
    drivers.sort(key=lambda d: abs(d["correlation"]), reverse=True)

    return jsonable({
        "store": store, "dept": dept,
        "seasonal_profile": prof.rename(columns={"Week_of_Year": "week"}),
        "top_weeks": top.rename(columns={"Week_of_Year": "week"}),
        "low_weeks": low.rename(columns={"Week_of_Year": "week"}),
        "holiday": {
            "box": box, "period_stats": period_stats, "events": events,
            "uplift_pct": round(uplift, 1), "holiday_n": n_hol,
            "holiday_share_pct": round(n_hol / total_n * 100.0, 1),
        },
        "markdown": markdown, "promo_comparison": promo_cmp, "top_promotions": top_promos,
        "markdown_intro_date": MARKDOWN_INTRO_DATE,
        "drivers": drivers,
        "elapsed_ms": _ms(t0),
    })


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def _clean_row(r: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "key": r["Model Key"], "name": r["Model"], "type": str(r["Type"]).lower(),
        "family": MODEL_FAMILY_LABEL.get(str(r["Type"]).lower(), str(r["Type"])),
        "method": r["Evaluation Method"], "folds": int(r["Successful Folds"]),
        "obs": int(r["Evaluated Obs"]),
        "wape": r["WAPE (%)"], "mae": r["MAE ($)"], "rmse": r["RMSE ($)"], "mape": r["MAPE (%)"],
        "status": r["Status"],
    }


def _rank(board: pd.DataFrame, metric_key: str) -> List[Dict[str, Any]]:
    ranked, _ = build_ranked_leaderboard(board, primary_metric=METRICS[metric_key])
    rows = []
    for _, r in ranked.iterrows():
        row = _clean_row(r.to_dict())
        row["rank"] = None if pd.isna(r.get("Rank")) else int(r["Rank"])
        rows.append(row)
    return rows


@lru_cache(maxsize=256)
def evaluate_one_model(store: int, dept: int, key: str) -> Dict[str, Any]:
    """Train + validate a single model (holdout and rolling-origin). Cached per series/model."""
    ctx = get_context(store, dept)
    catalog = get_all_models()
    if key not in catalog:
        raise ApiError(f"Unknown model '{key}'.", 404)

    t0 = time.perf_counter()
    hb, hp, test_df = evaluate_holdout(ctx.feat_df, models={key: catalog[key]})
    rb, rp = evaluate_rolling_origin(ctx.feat_df, horizon=CV_HORIZON, n_splits=CV_SPLITS,
                                     models={key: get_all_models()[key]})
    preds = hp.get(key)
    result: Dict[str, Any] = {
        "key": key,
        "holdout_board": hb,
        "cv_board": rb,
        "preds": preds,
        "test_df": test_df,
        "elapsed_ms": _ms(t0),
    }
    if preds is not None:
        res = analyze_residuals_and_holidays(test_df, preds)
        result["residual"] = res
        result["shock"] = evaluate_holdout_shocks_all_models(test_df, {key: preds}, get_all_models())
    return result


def _residual_payload(res: Dict[str, Any]) -> Dict[str, Any]:
    tbl = res["analysis_table"]
    return {
        "dates": _dates(tbl["Date"]),
        "actual": tbl["Actual"].tolist(),
        "predicted": tbl["Predicted"].tolist(),
        "error": tbl["Error"].tolist(),
        "mean": res["residual_mean"],
        "std": res["residual_std"],
        "top_errors": [
            {"date": pd.Timestamp(r["Date"]).strftime("%Y-%m-%d"), "actual": r["Actual"],
             "predicted": r["Predicted"], "error": r["Error"], "abs_error": r["Abs_Error"],
             "is_holiday": int(r["IsHoliday"])}
            for r in res["top_errors"]
        ],
        "holiday_stats": {
            k: {"n": v["Count"], "mean_actual": v["Mean_Actual"], "mean_predicted": v["Mean_Predicted"],
                "mae": v["MAE"], "rmse": v["RMSE"], "mape": v["MAPE"], "wape": v["WAPE"]}
            for k, v in res["holiday_stats"].items()
        },
    }


def evaluate_one_payload(store: int, dept: int, key: str) -> Dict[str, Any]:
    r = evaluate_one_model(store, dept, key)
    cv = _clean_row(r["cv_board"].iloc[0].to_dict())
    ho = _clean_row(r["holdout_board"].iloc[0].to_dict())
    return jsonable({
        "key": key, "cv": cv, "holdout": ho, "elapsed_ms": r["elapsed_ms"],
    })


def evaluate_all(store: int, dept: int, metric: str = "wape") -> Dict[str, Any]:
    """Full leaderboard payload: holdout + rolling-origin + residuals + shocks."""
    if metric not in METRICS:
        raise ApiError(f"Unknown metric '{metric}'. Use one of {list(METRICS)}.")
    t0 = time.perf_counter()
    ctx = get_context(store, dept)
    keys = list(get_all_models().keys())
    results = {k: evaluate_one_model(store, dept, k) for k in keys}

    hold_board = pd.concat([results[k]["holdout_board"] for k in keys], ignore_index=True)
    cv_board = pd.concat([results[k]["cv_board"] for k in keys], ignore_index=True)

    cv_rows = _rank(cv_board, metric)
    ho_rows = _rank(hold_board, metric)
    cv_winner = next((r for r in cv_rows if r["rank"] == 1), cv_rows[0])
    ho_winner = next((r for r in ho_rows if r["rank"] == 1), ho_rows[0])

    any_res = results[keys[0]]
    test_df = any_res["test_df"]

    predictions = {k: results[k]["preds"] for k in keys if results[k]["preds"] is not None}
    residuals = {k: _residual_payload(results[k]["residual"]) for k in predictions}
    shocks = pd.concat([results[k]["shock"] for k in predictions], ignore_index=True) \
        .sort_values("Overall WAPE (%)").reset_index(drop=True)
    shock_rows = [{
        "model": r["Model"], "overall_wape": r["Overall WAPE (%)"],
        "non_holiday_wape": r["Non-Holiday WAPE (%)"], "holiday_wape": r["Holiday Shock WAPE (%)"],
        "non_holiday_mae": r["Non-Holiday MAE ($)"], "holiday_mae": r["Holiday Shock MAE ($)"],
        "holiday_n": r["Holiday Weeks (N)"], "non_holiday_n": r["Non-Holiday Weeks (N)"],
    } for _, r in shocks.iterrows()]

    # baseline (original 4-lag regression) vs improved seasonal regression
    by_key = {r["key"]: r for r in cv_rows}
    base = by_key.get("lag_linear_regression")
    imp = by_key.get("improved_seasonal_regression")
    comparison = None
    if base and imp and base["wape"] and imp["wape"]:
        comparison = {
            "baseline_wape": base["wape"], "improved_wape": imp["wape"],
            "baseline_mae": base["mae"], "improved_mae": imp["mae"],
            "reduction_pct": round((base["wape"] - imp["wape"]) / base["wape"] * 100.0, 1),
        }

    n_hist = len(ctx.clean_history)
    return jsonable({
        "store": store, "dept": dept, "metric": metric, "metric_label": METRICS[metric],
        "model_rows": n_hist,
        "train_weeks": int(n_hist * 0.8),
        "test_weeks": int(len(test_df)),
        "cv_horizon": CV_HORIZON, "cv_folds": CV_SPLITS,
        "cv": cv_rows, "holdout": ho_rows,
        "winner": {"cv": cv_winner, "holdout": ho_winner},
        "holdout_test": {
            "dates": _dates(test_df["Date"]),
            "actual": test_df["Weekly_Sales"].tolist(),
            "is_holiday": test_df["IsHoliday"].astype(int).tolist(),
            "train_start": ctx.clean_history["Date"].iloc[0],
            "train_end": ctx.clean_history["Date"].iloc[int(n_hist * 0.8) - 1],
            "train_dates": _dates(ctx.clean_history["Date"].iloc[: int(n_hist * 0.8)]),
            "train_sales": ctx.clean_history["Weekly_Sales"].iloc[: int(n_hist * 0.8)].tolist(),
        },
        "predictions": predictions,
        "residuals": residuals,
        "shocks": shock_rows,
        "comparison": comparison,
        "elapsed_ms": _ms(t0),
    })


def champion_key(store: int, dept: int, metric: str = "wape") -> str:
    return evaluate_all(store, dept, metric)["winner"]["cv"]["key"]


# ---------------------------------------------------------------------------
# Future forecast
# ---------------------------------------------------------------------------

def _drivers(model, X: pd.DataFrame) -> List[Dict[str, Any]]:
    """Model-agnostic 'what drives this forecast' scores (normalised to 100 %)."""
    names = list(getattr(model, "feature_names", []) or [])
    inner = getattr(model, "model", None)
    if not names or inner is None:
        return []
    scores = None
    if hasattr(inner, "feature_importances_"):
        scores = np.asarray(inner.feature_importances_, dtype=float)
    elif hasattr(inner, "coef_"):
        scores = np.abs(np.asarray(inner.coef_, dtype=float)) * X[names].std().values
    if scores is None or scores.sum() <= 0:
        return []
    scores = scores / scores.sum() * 100.0
    out = [{"feature": n, "importance": float(s)} for n, s in zip(names, scores)]
    return sorted(out, key=lambda d: d["importance"], reverse=True)


def fit_model_on_history(ctx: SeriesContext, key: str):
    model = get_all_models()[key]
    hist = ctx.clean_history
    if model.model_type in ("ml", "baseline") and getattr(model, "feature_names", None):
        model.fit(y=hist["Weekly_Sales"], X=hist)
    else:
        model.fit(y=hist["Weekly_Sales"])
    return model


def calibrated_residual_std(store: int, dept: int, key: str) -> float:
    """One-step error scale from the model's own rolling-origin RMSE (fallback: 15 % of series std)."""
    ctx = get_context(store, dept)
    try:
        rmse = evaluate_one_model(store, dept, key)["cv_board"].iloc[0]["RMSE ($)"]
        if rmse is not None and not pd.isna(rmse) and rmse > 0:
            return float(rmse)
    except Exception:
        pass
    return float(ctx.clean_history["Weekly_Sales"].std() * 0.15)


def forecast_payload(store: int, dept: int, model_key: Optional[str], horizon: int,
                     metric: str = "wape") -> Dict[str, Any]:
    if not 1 <= horizon <= 26:
        raise ApiError("horizon must be between 1 and 26 weeks.")
    t0 = time.perf_counter()
    ctx = get_context(store, dept)
    champ = champion_key(store, dept, metric)
    key = champ if model_key in (None, "", "auto") else model_key
    if key not in get_all_models():
        raise ApiError(f"Unknown model '{key}'.", 404)

    model = fit_model_on_history(ctx, key)
    sigma = calibrated_residual_std(store, dept, key)
    last = pd.to_datetime(ctx.series_df["Date"].iloc[-1])
    fut_cal = ctx.calendar[ctx.calendar["Date"] > last]
    fut = generate_future_forecast(model, ctx.series_df, horizon=horizon, residual_std=sigma,
                                   z_score=1.96, future_features=fut_cal)
    covered = int(fut["Date"].isin(fut_cal["Date"].dt.strftime("%Y-%m-%d")).sum())
    hist = ctx.series_df.tail(26)

    rows = [{
        "date": r["Date"], "week": int(r["Week"]), "holiday": r["Holiday"], "is_holiday": int(r["IsHoliday"]),
        "forecast": r["Forecast ($)"], "lower": r["Lower_Bound ($)"], "upper": r["Upper_Bound ($)"],
    } for _, r in fut.iterrows()]

    return jsonable({
        "store": store, "dept": dept, "model": {"key": key, "name": model.name, "type": model.model_type,
                                                "description": model.description},
        "is_champion": key == champ, "champion_key": champ, "horizon": horizon,
        "forecast": rows,
        "history": {"dates": _dates(hist["Date"]), "sales": hist["Weekly_Sales"].astype(float).tolist()},
        "residual_std": sigma,
        "demand_std": float(ctx.clean_history["Weekly_Sales"].std()),
        "interval": "95% prediction interval, width grows with sqrt(horizon)",
        "known_calendar_weeks": covered,
        "drivers": _drivers(model, ctx.clean_history),
        "total_forecast": float(fut["Forecast ($)"].sum()),
        "elapsed_ms": _ms(t0),
    })


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

def inventory_payload(p: Dict[str, Any]) -> Dict[str, Any]:
    def num(name: str, default: float, lo: float = -1e12, hi: float = 1e12) -> float:
        try:
            v = float(p.get(name, default))
        except (TypeError, ValueError):
            raise ApiError(f"'{name}' must be a number.")
        if math.isnan(v) or v < lo or v > hi:
            raise ApiError(f"'{name}' is out of range.")
        return v

    forecast = num("forecast_weekly", 0.0, 0.0)
    std = num("demand_std", 0.0, 0.0)
    lead = int(num("lead_time_weeks", 2, 1, 26))
    review = int(num("review_period_weeks", 1, 1, 26))
    service = num("service_level_pct", 95.0, 50.0, 99.99)
    on_hand = num("current_on_hand", 0.0, 0.0)
    on_order = num("incoming_orders", 0.0, 0.0)
    back = num("backorders", 0.0, 0.0)
    unit_cost = num("unit_cost", 1.0, 0.0001)
    monetary = bool(p.get("is_monetary_scenario", True))

    kw = dict(forecast_weekly=forecast, demand_std=std, lead_time_weeks=lead, review_period_weeks=review,
              current_on_hand=on_hand, incoming_orders=on_order, backorders=back,
              unit_cost=unit_cost, is_monetary_scenario=monetary)
    result = calculate_inventory_policy(service_level_pct=service, **kw)

    sensitivity = []
    for lvl in [80.0, 85.0, 90.0, 95.0, 98.0, 99.0, 99.9]:
        r = calculate_inventory_policy(service_level_pct=lvl, **kw)
        sensitivity.append({"service_level": lvl, "z": r["z_score"], "safety_stock": r["safety_stock"],
                            "reorder_point": r["reorder_point"], "order_quantity": r["order_quantity"]})
    lead_sens = []
    for L in range(1, 9):
        kw2 = {**kw, "lead_time_weeks": L}
        r = calculate_inventory_policy(service_level_pct=service, **kw2)
        lead_sens.append({"lead_time": L, "safety_stock": r["safety_stock"], "reorder_point": r["reorder_point"],
                          "target_inventory": r["target_inventory"]})
    result["order_status"] = "Place order" if result["order_quantity"] > 0 else "Stock sufficient"
    return jsonable({"policy": result, "service_sensitivity": sensitivity, "lead_time_sensitivity": lead_sens})
