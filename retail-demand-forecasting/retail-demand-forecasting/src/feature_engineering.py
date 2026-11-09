"""
feature_engineering.py
Generates strictly non-leaking lag features, rolling statistics, calendar encodings,
and multi-step recursive forecasting feature states.
"""

from typing import List, Tuple, Optional
import pandas as pd
import numpy as np


LAG_STEPS = [1, 2, 3, 4, 8, 13, 52]
ROLLING_WINDOWS = [4, 8, 13]

FEATURE_SETS = {
    "lag_only": ["Lag_1", "Lag_2", "Lag_3", "Lag_4"],
    "seasonal": ["Lag_1", "Lag_2", "Lag_3", "Lag_4", "Lag_52", "Week_Sin", "Week_Cos", "IsHoliday"],
    "tree_all": [
        "Lag_1", "Lag_2", "Lag_3", "Lag_4", "Lag_8", "Lag_13", "Lag_52",
        "Rolling_Mean_4", "Rolling_Std_4", "Rolling_Mean_8", "Rolling_Mean_13",
        "Week_Sin", "Week_Cos", "Month", "Quarter", "IsHoliday",
        "Temperature", "Fuel_Price", "CPI", "Unemployment", "Total_MarkDown"
    ]
}


def add_lag_features(df: pd.DataFrame, target_col: str = "Weekly_Sales", lags: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Creates historical lag features for the specified target column.
    """
    lags = lags or LAG_STEPS
    res = df.copy()
    n_rows = len(res)

    for lag in lags:
        if n_rows > lag:
            res[f"Lag_{lag}"] = res[target_col].shift(lag)
        else:
            # If dataset is shorter than the lag (e.g. lag 52 on a 30-week series), do not create it
            pass

    return res


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "Weekly_Sales",
    windows: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Calculates rolling mean, std, min, and max over specified windows.
    IMPORTANT: We shift the target by 1 step before computing rolling statistics
    to guarantee that rolling windows strictly use past data and prevent future leakage.
    """
    windows = windows or ROLLING_WINDOWS
    res = df.copy()

    # Strictly shift by 1 to prevent data leakage of the current week's target
    shifted_target = res[target_col].shift(1)

    for w in windows:
        if len(res) >= w:
            res[f"Rolling_Mean_{w}"] = shifted_target.rolling(window=w, min_periods=max(1, w // 2)).mean()
            res[f"Rolling_Std_{w}"] = shifted_target.rolling(window=w, min_periods=max(2, w // 2)).std().fillna(0.0)
            res[f"Rolling_Min_{w}"] = shifted_target.rolling(window=w, min_periods=1).min()
            res[f"Rolling_Max_{w}"] = shifted_target.rolling(window=w, min_periods=1).max()

    return res


def build_feature_dataframe(
    df: pd.DataFrame,
    target_col: str = "Weekly_Sales",
    include_lag_52: bool = True
) -> pd.DataFrame:
    """
    Builds the complete feature-engineered dataset:
    - Lag features (1, 2, 3, 4, 8, 13, 52)
    - Rolling stats (mean, std, min, max) on shifted target
    - Calendar & cyclical features (if not already present)
    """
    res = df.copy()

    # Determine applicable lags
    lags = [1, 2, 3, 4, 8, 13]
    if include_lag_52 and len(res) > 52:
        lags.append(52)

    res = add_lag_features(res, target_col=target_col, lags=lags)
    res = add_rolling_features(res, target_col=target_col, windows=ROLLING_WINDOWS)

    return res


def get_available_features(df: pd.DataFrame, requested_features: List[str]) -> List[str]:
    """
    Filters requested features to only those that exist in the DataFrame without all-NaN values.
    """
    available = []
    for f in requested_features:
        if f in df.columns and not df[f].isnull().all():
            available.append(f)
    return available


def update_recursive_features(
    historical_sales: List[float],
    next_date: pd.Timestamp,
    is_holiday: int,
    known_features: Optional[dict] = None
) -> dict:
    """
    Builds a single-step feature dictionary for future recursive forecasting without cheating.
    
    Parameters:
    - historical_sales: list of past sales [..., y_t-2, y_t-1, y_t] (including previous step forecasts)
    - next_date: datetime for the prediction week
    - is_holiday: 0 or 1
    - known_features: optional external features (Temperature, CPI, etc., if available or estimated)
    """
    feats = {}
    hist = np.array(historical_sales)
    n = len(hist)

    # Lags
    feats["Lag_1"] = float(hist[-1]) if n >= 1 else 0.0
    feats["Lag_2"] = float(hist[-2]) if n >= 2 else feats["Lag_1"]
    feats["Lag_3"] = float(hist[-3]) if n >= 3 else feats["Lag_2"]
    feats["Lag_4"] = float(hist[-4]) if n >= 4 else feats["Lag_3"]
    feats["Lag_8"] = float(hist[-8]) if n >= 8 else feats["Lag_4"]
    feats["Lag_13"] = float(hist[-13]) if n >= 13 else feats["Lag_8"]
    feats["Lag_52"] = float(hist[-52]) if n >= 52 else feats["Lag_1"]

    # Rolling statistics over available history
    for w in ROLLING_WINDOWS:
        sub = hist[-w:] if n >= w else hist
        feats[f"Rolling_Mean_{w}"] = float(np.mean(sub))
        feats[f"Rolling_Std_{w}"] = float(np.std(sub)) if len(sub) > 1 else 0.0
        feats[f"Rolling_Min_{w}"] = float(np.min(sub))
        feats[f"Rolling_Max_{w}"] = float(np.max(sub))

    # Calendar features
    week_of_year = int(next_date.isocalendar().week)
    month = int(next_date.month)
    quarter = int(next_date.quarter)

    feats["Week_of_Year"] = week_of_year
    feats["Month"] = month
    feats["Quarter"] = quarter
    feats["Week_Sin"] = float(np.sin(2 * np.pi * week_of_year / 52.0))
    feats["Week_Cos"] = float(np.cos(2 * np.pi * week_of_year / 52.0))
    feats["Month_Sin"] = float(np.sin(2 * np.pi * month / 12.0))
    feats["Month_Cos"] = float(np.cos(2 * np.pi * month / 12.0))
    feats["IsHoliday"] = int(is_holiday)

    # Defaults for external features
    defaults = {
        "Temperature": 60.0,
        "Fuel_Price": 3.5,
        "CPI": 210.0,
        "Unemployment": 7.5,
        "Total_MarkDown": 0.0,
        "MarkDown1": 0.0,
        "MarkDown2": 0.0,
        "MarkDown3": 0.0,
        "MarkDown4": 0.0,
        "MarkDown5": 0.0,
        "MarkDown_Available": 1,
        "Size": 150000
    }

    if known_features:
        defaults.update(known_features)

    for k, v in defaults.items():
        if k not in feats:
            feats[k] = v

    return feats
