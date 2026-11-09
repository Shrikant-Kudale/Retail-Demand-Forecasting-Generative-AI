"""
data_loader.py
Loads, merges, and validates Walmart retail datasets with configurable fallback paths.
"""

import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_CANDIDATE_DIRS = [
    os.path.join(PROJECT_ROOT, "data"),
    "data",
    os.path.join("..", "data"),
    ".",
]


def find_data_file(filename: str, candidate_dirs: Optional[List[str]] = None) -> str:
    """
    Search for a data file across standard candidate directories.
    Raises FileNotFoundError if file is not located.
    """
    search_paths = candidate_dirs or DEFAULT_CANDIDATE_DIRS
    for d in search_paths:
        target = os.path.join(d, filename)
        if os.path.exists(target):
            return os.path.abspath(target)
    
    raise FileNotFoundError(
        f"Unable to locate '{filename}'. Searched paths: {search_paths}. "
        "Please specify data_dir explicitly."
    )


def load_raw_datasets(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Loads train.csv, features.csv, stores.csv, and test.csv from data_dir or candidate paths.
    """
    candidate_dirs = [data_dir] if data_dir else DEFAULT_CANDIDATE_DIRS

    train_path = find_data_file("train.csv", candidate_dirs)
    features_path = find_data_file("features.csv", candidate_dirs)
    stores_path = find_data_file("stores.csv", candidate_dirs)
    
    train = pd.read_csv(train_path)
    features = pd.read_csv(features_path)
    stores = pd.read_csv(stores_path)

    test = None
    try:
        test_path = find_data_file("test.csv", candidate_dirs)
        test = pd.read_csv(test_path)
    except FileNotFoundError:
        pass

    return train, features, stores, test


def prepare_merged_dataset(
    train: pd.DataFrame,
    features: pd.DataFrame,
    stores: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges train, features, and stores data on Store, Date, and Store keys.
    Converts Date to datetime and sorts chronologically.
    """
    train_clean = train.copy()
    features_clean = features.copy()
    stores_clean = stores.copy()

    # Convert dates to datetime
    train_clean["Date"] = pd.to_datetime(train_clean["Date"])
    features_clean["Date"] = pd.to_datetime(features_clean["Date"])

    # Avoid duplicate IsHoliday column during merge
    if "IsHoliday" in features_clean.columns and "IsHoliday" in train_clean.columns:
        features_clean = features_clean.drop(columns=["IsHoliday"])

    # Merge sales with features on [Store, Date]
    merged = train_clean.merge(features_clean, on=["Store", "Date"], how="left")

    # Merge with store attributes on [Store]
    merged = merged.merge(stores_clean, on="Store", how="left")

    # Ensure chronological sorting
    merged = merged.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

    return merged


def validate_dataset(df: pd.DataFrame) -> Dict[str, any]:
    """
    Performs comprehensive data quality checks on the merged dataset:
    - Row count and column types
    - Duplicate keys check
    - Missing dates check
    - Negative sales count (returns)
    - Missing values breakdown
    """
    report = {}
    report["total_rows"] = len(df)
    report["date_min"] = df["Date"].min()
    report["date_max"] = df["Date"].max()
    report["unique_stores"] = df["Store"].nunique()
    report["unique_depts"] = df["Dept"].nunique()
    
    # Duplicate check on [Store, Dept, Date]
    duplicates = df.duplicated(subset=["Store", "Dept", "Date"]).sum()
    report["duplicate_keys"] = int(duplicates)

    # Missing target check
    missing_targets = df["Weekly_Sales"].isnull().sum()
    report["missing_sales"] = int(missing_targets)

    # Negative sales check (returns/adjustments)
    neg_sales = (df["Weekly_Sales"] < 0).sum()
    report["negative_sales_count"] = int(neg_sales)
    report["negative_sales_pct"] = float(neg_sales / len(df) * 100) if len(df) > 0 else 0.0

    # Missing values by column
    missing_counts = df.isnull().sum()
    report["missing_by_column"] = missing_counts[missing_counts > 0].to_dict()

    return report


def get_series(
    df: pd.DataFrame,
    store_id: int = 1,
    dept_id: int = 1,
    min_observations: int = 52
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Extracts, validates, and sorts the time series for a specific Store and Department.
    Detects missing weekly gaps and returns the series and diagnostic metadata.
    """
    series_df = df[(df["Store"] == store_id) & (df["Dept"] == dept_id)].copy()
    series_df = series_df.sort_values("Date").reset_index(drop=True)

    diagnostics = {
        "store": store_id,
        "dept": dept_id,
        "observations": len(series_df),
        "is_sufficient": len(series_df) >= min_observations,
        "has_gaps": False,
        "gap_count": 0,
        "start_date": None,
        "end_date": None,
        "warning": None
    }

    if len(series_df) == 0:
        diagnostics["warning"] = f"No records found for Store {store_id}, Department {dept_id}."
        return series_df, diagnostics

    diagnostics["start_date"] = series_df["Date"].min()
    diagnostics["end_date"] = series_df["Date"].max()

    if len(series_df) < min_observations:
        diagnostics["warning"] = (
            f"Series has only {len(series_df)} observations. Minimum required is {min_observations}."
        )

    # Check for weekly frequency gaps
    date_diffs = series_df["Date"].diff().dropna()
    irregular_diffs = date_diffs[date_diffs != pd.Timedelta(days=7)]
    if len(irregular_diffs) > 0:
        diagnostics["has_gaps"] = True
        diagnostics["gap_count"] = len(irregular_diffs)
        diagnostics["warning"] = f"Series contains {len(irregular_diffs)} non-7-day step gaps."

    return series_df, diagnostics


def get_available_stores_and_depts(df: pd.DataFrame) -> Tuple[List[int], Dict[int, List[int]]]:
    """
    Returns sorted list of stores and a mapping from store_id to available department IDs.
    """
    stores = sorted(df["Store"].unique().tolist())
    dept_map = {}
    for store in stores:
        depts = sorted(df[df["Store"] == store]["Dept"].unique().tolist())
        dept_map[store] = depts
    return stores, dept_map
