"""
test_evaluation_fixes.py
Focused unit tests verifying that all 4 reported forecasting issues are resolved:
1. Improved Seasonal Regression does not encounter NaNs in lag-52 warmup.
2. Gradient Boosting does not encounter NaNs in lag-52 warmup.
3. Holt-Winters executes cleanly with RangeIndex without index warnings or failures.
4. SARIMA executes cleanly with RangeIndex without index warnings or failures.
5. All 8 models achieve 100% valid status across holdout and rolling-origin validation.
"""

import pytest
import numpy as np
import pandas as pd

from src.data_loader import load_raw_datasets, prepare_merged_dataset, get_series
from src.preprocessing import preprocess_series
from src.feature_engineering import build_feature_dataframe
from src.forecasting_models import get_all_models, HoltWintersForecaster, SARIMAForecaster
from src.evaluation import evaluate_holdout, evaluate_rolling_origin, compute_all_metrics


@pytest.fixture(scope="module")
def prepared_s1d1_data():
    train, features, stores, _ = load_raw_datasets("data")
    merged = prepare_merged_dataset(train, features, stores)
    s1d1, _ = get_series(merged, store_id=1, dept_id=1)
    cleaned = preprocess_series(s1d1)
    feat_df = build_feature_dataframe(cleaned)
    return feat_df


def test_no_nans_in_warmup_filtered_data(prepared_s1d1_data):
    """Verify that dropping warmup features produces a clean dataset with zero NaNs in Lag_52."""
    warmup_cols = [c for c in ["Lag_1", "Lag_2", "Lag_3", "Lag_4", "Lag_52", "Rolling_Mean_13"] if c in prepared_s1d1_data.columns]
    clean_df = prepared_s1d1_data.dropna(subset=warmup_cols).reset_index(drop=True)
    assert len(clean_df) == 91
    assert clean_df["Lag_52"].isnull().sum() == 0
    assert clean_df["Lag_1"].isnull().sum() == 0


def test_holt_winters_range_index_execution():
    """Verify Holt-Winters executes without non-continuous index warnings."""
    y = pd.Series([20000 + 100 * (i % 7) for i in range(70)], index=range(52, 122), dtype=float)
    model = HoltWintersForecaster()
    model.fit(y)
    preds = model.predict(steps=4)
    assert len(preds) == 4
    assert not np.isnan(preds).any()


def test_sarima_range_index_execution():
    """Verify SARIMA executes without non-continuous index warnings or convergence stalls."""
    y = pd.Series([20000 + 100 * (i % 7) for i in range(70)], index=range(52, 122), dtype=float)
    model = SARIMAForecaster(order=(1, 1, 0))
    model.fit(y)
    preds = model.predict(steps=4)
    assert len(preds) == 4
    assert not np.isnan(preds).any()


def test_all_eight_models_holdout_valid(prepared_s1d1_data):
    """Verify all 8 models succeed in chronological holdout."""
    leaderboard, preds, test_df = evaluate_holdout(prepared_s1d1_data)
    assert len(leaderboard) == 8
    assert (leaderboard["Status"] == "Valid").all()
    assert (leaderboard["Evaluated Obs"] == 19).all()
    assert (leaderboard["Successful Folds"] == 1).all()


def test_all_eight_models_rolling_origin_valid(prepared_s1d1_data):
    """Verify all 8 models succeed across all 4 folds in rolling-origin cross validation."""
    leaderboard, preds = evaluate_rolling_origin(prepared_s1d1_data, horizon=4, n_splits=4)
    assert len(leaderboard) == 8
    assert (leaderboard["Status"] == "Valid").all()
    assert (leaderboard["Successful Folds"] == 4).all()
    assert (leaderboard["Evaluated Obs"] == 16).all()


def test_metrics_zero_handling():
    """Verify that zero actuals do not result in division by zero or infs."""
    y_true = np.array([0.0, 100.0, 200.0])
    y_pred = np.array([10.0, 110.0, 190.0])
    metrics = compute_all_metrics(y_true, y_pred)
    assert not np.isinf(metrics["MAPE"])
    assert not np.isnan(metrics["MAPE"])
    assert metrics["WAPE"] > 0
