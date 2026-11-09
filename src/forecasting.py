"""
forecasting.py
Multi-step future forecasting engine with recursive feature state progression
and residual-calibrated prediction intervals.

Future weeks use the *known* calendar and covariates shipped with the dataset
(holiday flags, temperature, fuel price, CPI, unemployment from ``features.csv``)
whenever they are available, and fall back to a week-of-year holiday rule and the
last observed values otherwise. Sales themselves are never peeked at: every
predicted week is fed back as the next lag (recursive strategy).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.forecasting_models import BaseForecaster
from src.feature_engineering import update_recursive_features


# Standard US retail holidays in the Walmart calendar (ISO week -> event name)
KNOWN_HOLIDAY_WEEKS = {
    6: "Super Bowl",
    36: "Labor Day",
    47: "Thanksgiving",
    51: "Christmas",
    52: "Christmas / New Year",
}

EXTERNAL_COLS = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]
MARKDOWN_COLS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]


def is_walmart_holiday(dt: pd.Timestamp) -> Tuple[int, str]:
    """Rule-based fallback: is the given Friday a major Walmart holiday week?"""
    iso_week = int(dt.isocalendar().week)
    if iso_week in KNOWN_HOLIDAY_WEEKS:
        return 1, KNOWN_HOLIDAY_WEEKS[iso_week]
    return 0, "Regular Week"


def holiday_for_date(
    dt: pd.Timestamp,
    calendar: Optional[pd.DataFrame] = None,
) -> Tuple[int, str]:
    """
    Resolve the holiday flag for a future week.
    Uses the dataset's own calendar when the date is covered, else the ISO-week rule.
    """
    if calendar is not None and len(calendar) > 0:
        hit = calendar[calendar["Date"] == dt]
        if len(hit) > 0 and "IsHoliday" in hit.columns:
            flagged = int(bool(hit["IsHoliday"].iloc[0]))
            if flagged:
                week = int(dt.isocalendar().week)
                return 1, KNOWN_HOLIDAY_WEEKS.get(week, "Holiday Week")
            return 0, "Regular Week"
    return is_walmart_holiday(dt)


def _known_features_for(
    dt: pd.Timestamp,
    calendar: Optional[pd.DataFrame],
    last_row: Optional[pd.Series],
) -> Dict[str, float]:
    """Known/estimated external covariates for a future date."""
    known: Dict[str, float] = {}

    # 1) last observed values as a sensible carry-forward default
    if last_row is not None:
        for col in EXTERNAL_COLS:
            if col in last_row.index and pd.notna(last_row[col]):
                known[col] = float(last_row[col])
        if "Size" in last_row.index and pd.notna(last_row["Size"]):
            known["Size"] = float(last_row["Size"])

    # 2) dataset calendar overrides when the date is covered
    if calendar is not None and len(calendar) > 0:
        hit = calendar[calendar["Date"] == dt]
        if len(hit) > 0:
            row = hit.iloc[0]
            for col in EXTERNAL_COLS:
                if col in row.index and pd.notna(row[col]):
                    known[col] = float(row[col])
            md_total = 0.0
            for col in MARKDOWN_COLS:
                if col in row.index and pd.notna(row[col]):
                    known[col] = float(row[col])
                    md_total += float(row[col])
            known["Total_MarkDown"] = md_total
    return known


def generate_future_forecast(
    model: BaseForecaster,
    history_df: pd.DataFrame,
    horizon: int = 8,
    residual_std: Optional[float] = None,
    z_score: float = 1.96,
    future_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Generate multi-step sales forecasts beyond the end of ``history_df``.

    Parameters
    ----------
    model : fitted BaseForecaster
    history_df : DataFrame with ``Date`` and ``Weekly_Sales`` (and optionally the
        covariate columns, used as carry-forward defaults).
    horizon : number of weeks ahead.
    residual_std : empirical std of one-step forecast errors (for the interval).
    z_score : interval multiplier (1.96 -> 95 %).
    future_features : optional per-date known calendar/covariates
        (``Date``, ``IsHoliday``, ``Temperature``, ... ``MarkDown1-5``).

    Returns
    -------
    DataFrame with dates, point forecast and lower / upper bounds.
    """
    history_clean = history_df.sort_values("Date").reset_index(drop=True)
    last_date = pd.to_datetime(history_clean["Date"].iloc[-1])
    sales_history = history_clean["Weekly_Sales"].values.tolist()
    last_row = history_clean.iloc[-1]

    if residual_std is None or residual_std <= 0:
        residual_std = float(np.std(np.diff(sales_history))) if len(sales_history) > 2 else 1000.0

    calendar = None
    if future_features is not None and len(future_features) > 0:
        calendar = future_features.copy()
        calendar["Date"] = pd.to_datetime(calendar["Date"])

    future_dates = [last_date + pd.Timedelta(weeks=step) for step in range(1, horizon + 1)]
    records = []

    def _record(h: int, dt: pd.Timestamp, pred: float, hol_flag: int, hol_name: str) -> dict:
        p_val = max(0.0, float(pred))
        radius = z_score * residual_std * np.sqrt(h)   # uncertainty grows with sqrt(h)
        return {
            "Date": dt.strftime("%Y-%m-%d"),
            "Week": int(dt.isocalendar().week),
            "Holiday": hol_name,
            "IsHoliday": hol_flag,
            "Forecast ($)": round(p_val, 2),
            "Lower_Bound ($)": round(max(0.0, p_val - radius), 2),
            "Upper_Bound ($)": round(p_val + radius, 2),
            "Model": model.name,
        }

    if model.model_type == "statistical":
        # Time-series models forecast the whole horizon directly
        raw_preds = model.predict(steps=horizon)
        for h, (dt, pred) in enumerate(zip(future_dates, raw_preds), start=1):
            hol_flag, hol_name = holiday_for_date(dt, calendar)
            records.append(_record(h, dt, pred, hol_flag, hol_name))
    else:
        # Recursive strategy for feature-based models
        current_history = list(sales_history)
        for h, dt in enumerate(future_dates, start=1):
            hol_flag, hol_name = holiday_for_date(dt, calendar)
            known = _known_features_for(dt, calendar, last_row)
            feat_dict = update_recursive_features(
                historical_sales=current_history,
                next_date=dt,
                is_holiday=hol_flag,
                known_features=known,
            )
            pred = max(0.0, float(model.predict(steps=1, X=pd.DataFrame([feat_dict]))[0]))
            records.append(_record(h, dt, pred, hol_flag, hol_name))
            current_history.append(pred)   # predicted value becomes the next lag

    return pd.DataFrame(records)
