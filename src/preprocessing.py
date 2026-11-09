"""
preprocessing.py
Data cleaning, gap handling, markdown treatment, and calendar decomposition.
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np


# In the Walmart dataset, promotional MarkDowns were introduced in November 2011.
MARKDOWN_INTRO_DATE = pd.to_datetime("2011-11-04")
MARKDOWN_COLS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]


def clean_series(
    df: pd.DataFrame,
    handle_negative: str = "keep",
    regularize_gaps: bool = False
) -> pd.DataFrame:
    """
    Cleans a single store-dept time series.
    
    Parameters:
    - df: DataFrame containing time series observations
    - handle_negative: 'keep' (default, retains raw values),
                       'clip_zero' (replaces negative sales with 0, representing zero net sales),
                       'flag' (adds IsNegative flag).
    - regularize_gaps: If True, reindexes the series to a complete weekly frequency (freq='W-FRI').
    """
    clean_df = df.copy()
    clean_df["Date"] = pd.to_datetime(clean_df["Date"])
    clean_df = clean_df.sort_values("Date").reset_index(drop=True)

    # Negative sales handling
    clean_df["IsNegativeReturn"] = (clean_df["Weekly_Sales"] < 0).astype(int)
    if handle_negative == "clip_zero":
        clean_df["Weekly_Sales"] = clean_df["Weekly_Sales"].clip(lower=0.0)

    # Regularize weekly gaps if requested
    if regularize_gaps and len(clean_df) > 1:
        min_date = clean_df["Date"].min()
        max_date = clean_df["Date"].max()
        full_idx = pd.date_range(min_date, max_date, freq="W-FRI")
        clean_df = clean_df.set_index("Date").reindex(full_idx)
        clean_df.index.name = "Date"
        clean_df = clean_df.reset_index()

        # Interpolate or fill missing dates created by regularizing
        clean_df["Weekly_Sales"] = clean_df["Weekly_Sales"].interpolate(method="linear")
        for col in ["Store", "Dept", "Type", "Size"]:
            if col in clean_df.columns:
                clean_df[col] = clean_df[col].ffill().bfill()

    return clean_df


def handle_markdown_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles promotional markdown features with domain rigor:
    - Before Nov 2011: Markdowns were unrecorded/uncollected by Walmart.
    - After Nov 2011: Null values indicate no promotional markdown (confirmed 0).
    - Creates 'MarkDown_Available' indicator (0 prior to Nov 2011, 1 afterwards).
    - Fills NaN markdowns with 0.0.
    - Creates 'Total_MarkDown' summing MarkDown1 through MarkDown5.
    """
    res = df.copy()
    
    # MarkDown availability flag
    res["MarkDown_Available"] = (res["Date"] >= MARKDOWN_INTRO_DATE).astype(int)

    # Impute missing markdown values to 0.0
    for col in MARKDOWN_COLS:
        if col in res.columns:
            res[col] = res[col].fillna(0.0)
        else:
            res[col] = 0.0

    # Total markdown value
    res["Total_MarkDown"] = res[MARKDOWN_COLS].sum(axis=1)
    res["Has_Active_MarkDown"] = (res["Total_MarkDown"] > 0).astype(int)

    return res


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts standard calendar components and cyclical weekly trigonometric features.
    """
    res = df.copy()
    res["Date"] = pd.to_datetime(res["Date"])
    
    res["Year"] = res["Date"].dt.year
    res["Month"] = res["Date"].dt.month
    res["Quarter"] = res["Date"].dt.quarter
    # ISO week of year (1-53)
    res["Week_of_Year"] = res["Date"].dt.isocalendar().week.astype(int)

    # Cyclical week encoding (captures annual recurrence smoothly)
    res["Week_Sin"] = np.sin(2 * np.pi * res["Week_of_Year"] / 52.0)
    res["Week_Cos"] = np.cos(2 * np.pi * res["Week_of_Year"] / 52.0)

    # Month cyclical encoding
    res["Month_Sin"] = np.sin(2 * np.pi * res["Month"] / 12.0)
    res["Month_Cos"] = np.cos(2 * np.pi * res["Month"] / 12.0)

    # Holiday boolean to integer
    if "IsHoliday" in res.columns:
        res["IsHoliday"] = res["IsHoliday"].astype(int)

    return res


def preprocess_series(
    df: pd.DataFrame,
    handle_negative: str = "keep"
) -> pd.DataFrame:
    """
    Unified end-to-end preprocessing pipeline for a selected time series:
    1. Clean series & handle returns
    2. Handle markdowns
    3. Add calendar and cyclical features
    4. Fill any remaining economic feature NaNs with forward/backward fill
    """
    df_clean = clean_series(df, handle_negative=handle_negative)
    df_promo = handle_markdown_features(df_clean)
    df_calendar = add_calendar_features(df_promo)

    # Impute CPI, Unemployment, Fuel_Price, Temperature if NaNs exist
    for col in ["CPI", "Unemployment", "Fuel_Price", "Temperature"]:
        if col in df_calendar.columns:
            df_calendar[col] = df_calendar[col].ffill().bfill()

    return df_calendar
