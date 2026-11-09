"""
test_forecasting.py
Comprehensive automated test suite covering:
- Data loading, multi-path fallback, joins, duplicate detection, and chronological ordering.
- Non-leaking lag and rolling feature calculations (strictly shifted target).
- Metric robustness (MAE, RMSE, MAPE with zero actuals, WAPE).
- Chronological train-test separation without shuffling.
- 8-model execution, leaderboard generation, and programmatic winner selection.
- Multi-step recursive forecasting without future-actual leakage.
- Inventory decision support (monetary planning and demonstration unit scenario).
- Edge-case handling for insufficient series length and missing features.
"""

import pytest
import numpy as np
import pandas as pd

from src.data_loader import (
    load_raw_datasets,
    prepare_merged_dataset,
    validate_dataset,
    get_series,
    get_available_stores_and_depts,
)
from src.preprocessing import clean_series, handle_markdown_features, add_calendar_features, preprocess_series
from src.feature_engineering import (
    add_lag_features,
    add_rolling_features,
    build_feature_dataframe,
    update_recursive_features,
)
from src.forecasting_models import (
    get_all_models,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    HoltWintersForecaster,
    SARIMAForecaster,
    LagLinearRegressionForecaster,
    ImprovedSeasonalRegressionForecaster,
    RandomForestForecaster,
    GradientBoostingForecaster,
)
from src.evaluation import (
    calculate_mae,
    calculate_rmse,
    calculate_mape,
    calculate_wape,
    compute_all_metrics,
    evaluate_holdout,
    evaluate_rolling_origin,
    build_ranked_leaderboard,
    analyze_residuals_and_holidays,
    evaluate_holdout_shocks_all_models,
)
from src.forecasting import generate_future_forecast, is_walmart_holiday
from src.inventory import calculate_inventory_policy, get_z_score


# ---------------------------------------------------------
# 1. DATA LOADING & VALIDATION TESTS
# ---------------------------------------------------------

@pytest.fixture(scope="module")
def loaded_raw_data():
    return load_raw_datasets("data")


@pytest.fixture(scope="module")
def merged_dataset(loaded_raw_data):
    train, features, stores, _ = loaded_raw_data
    return prepare_merged_dataset(train, features, stores)


def test_data_loading_and_joins(loaded_raw_data, merged_dataset):
    train, features, stores, test = loaded_raw_data
    assert len(train) > 0
    assert len(features) > 0
    assert len(stores) > 0
    assert len(merged_dataset) == len(train)
    assert "Temperature" in merged_dataset.columns
    assert "Type" in merged_dataset.columns
    assert "Size" in merged_dataset.columns


def test_chronological_sorting_and_keys(merged_dataset):
    report = validate_dataset(merged_dataset)
    assert report["duplicate_keys"] == 0
    assert report["total_rows"] == len(merged_dataset)
    assert report["missing_sales"] == 0

    # Ensure Date is monotonic per Store & Dept
    s1d1 = merged_dataset[(merged_dataset["Store"] == 1) & (merged_dataset["Dept"] == 1)]
    assert s1d1["Date"].is_monotonic_increasing


def test_store_and_dept_discovery(merged_dataset):
    stores, dept_map = get_available_stores_and_depts(merged_dataset)
    assert 1 in stores
    assert len(dept_map[1]) > 0
    assert 1 in dept_map[1]


# ---------------------------------------------------------
# 2. FEATURE ENGINEERING & LEAKAGE PREVENTION TESTS
# ---------------------------------------------------------

def test_lag_feature_generation():
    df = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=10, freq="W-FRI"),
        "Weekly_Sales": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
    })
    lagged = add_lag_features(df, lags=[1, 2, 4])
    assert "Lag_1" in lagged.columns
    assert np.isnan(lagged["Lag_1"].iloc[0])
    assert lagged["Lag_1"].iloc[1] == 100
    assert lagged["Lag_2"].iloc[2] == 100
    assert lagged["Lag_4"].iloc[4] == 100


def test_rolling_feature_strictly_prevents_lookahead_leakage():
    """
    CRITICAL: Rolling features must shift target by 1 step before computing rolling statistics.
    The current week's target must NEVER influence the current week's rolling statistics.
    """
    df = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=10, freq="W-FRI"),
        "Weekly_Sales": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    })
    rolled = add_rolling_features(df, windows=[4])
    
    # At index 4, past 4 values were [10, 20, 30, 40] (mean = 25.0).
    # Current value at index 4 is 50.0. If leaking, mean would be (20+30+40+50)/4 = 35.0.
    assert rolled["Rolling_Mean_4"].iloc[4] == 25.0, "Rolling feature leaked current observation!"
    assert rolled["Rolling_Mean_4"].iloc[4] != 35.0


def test_calendar_and_markdown_features():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2011-05-06", "2012-05-04"]),
        "Weekly_Sales": [20000.0, 22000.0],
        "MarkDown1": [np.nan, 500.0],
        "IsHoliday": [0, 0]
    })
    pre = preprocess_series(df)
    assert "Week_Sin" in pre.columns
    assert "Week_Cos" in pre.columns
    assert "MarkDown_Available" in pre.columns
    # Pre-Nov 2011 was 0, Post-Nov 2011 is 1
    assert pre["MarkDown_Available"].iloc[0] == 0
    assert pre["MarkDown_Available"].iloc[1] == 1
    assert pre["MarkDown1"].iloc[0] == 0.0


# ---------------------------------------------------------
# 3. METRIC CORRECTNESS & ZERO-DENOMINATOR HANDLING
# ---------------------------------------------------------

def test_metrics_accuracy():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 330.0])

    mae = calculate_mae(y_true, y_pred)
    assert mae == pytest.approx(16.666, rel=1e-2)

    rmse = calculate_rmse(y_true, y_pred)
    # sqrt((100 + 100 + 900) / 3) = sqrt(1100 / 3) = 19.1485
    assert rmse == pytest.approx(19.1485, rel=1e-2)

    wape = calculate_wape(y_true, y_pred)
    # WAPE = sum(10+10+30) / sum(100+200+300) = 50 / 600 = 8.33%
    assert wape == pytest.approx(8.333, rel=1e-2)


def test_metrics_safe_with_zero_actuals():
    y_true = np.array([0.0, 0.0, 100.0])
    y_pred = np.array([10.0, 5.0, 105.0])

    # Should not raise ZeroDivisionError or return inf
    mape = calculate_mape(y_true, y_pred)
    assert not np.isinf(mape)
    assert not np.isnan(mape)

    wape = calculate_wape(y_true, y_pred)
    assert not np.isinf(wape)
    assert wape > 0


# ---------------------------------------------------------
# 4. CHRONOLOGICAL SPLIT & 8-MODEL LEADERBOARD
# ---------------------------------------------------------

def test_all_eight_models_holdout_and_ranking(merged_dataset):
    s1d1, _ = get_series(merged_dataset, store_id=1, dept_id=1)
    cleaned = preprocess_series(s1d1)
    feat_df = build_feature_dataframe(cleaned)

    leaderboard, preds, test_df = evaluate_holdout(feat_df, train_ratio=0.8)
    assert len(leaderboard) == 8
    assert (leaderboard["Status"] == "Valid").all()

    ranked, winner = build_ranked_leaderboard(leaderboard, primary_metric="WAPE (%)")
    assert len(ranked) == 8
    assert winner in ["improved_seasonal_regression", "random_forest", "gradient_boosting", "seasonal_naive"]
    assert ranked["Rank"].iloc[0] == 1.0


def test_rolling_origin_cross_validation(merged_dataset):
    s1d1, _ = get_series(merged_dataset, store_id=1, dept_id=1)
    cleaned = preprocess_series(s1d1)
    feat_df = build_feature_dataframe(cleaned)

    ro_board, ro_preds = evaluate_rolling_origin(feat_df, horizon=4, n_splits=4)
    assert len(ro_board) == 8
    assert (ro_board["Status"] == "Valid").all()
    assert (ro_board["Successful Folds"] == 4).all()
    assert (ro_board["Evaluated Obs"] == 16).all()


# ---------------------------------------------------------
# 5. FUTURE FORECASTING WITHOUT FUTURE-ACTUAL LEAKAGE
# ---------------------------------------------------------

def test_future_forecast_generation_and_dates(merged_dataset):
    s1d1, _ = get_series(merged_dataset, store_id=1, dept_id=1)
    last_historical_date = s1d1["Date"].max()
    
    models = get_all_models()
    model = models["naive"]
    model.fit(s1d1["Weekly_Sales"])

    forecast_df = generate_future_forecast(model, s1d1, horizon=8)
    assert len(forecast_df) == 8
    first_forecast_date = pd.to_datetime(forecast_df["Date"].iloc[0])
    
    # Must start exactly 7 days after the last historical date
    assert first_forecast_date == last_historical_date + pd.Timedelta(days=7)
    assert (forecast_df["Forecast ($)"] >= 0).all()
    assert (forecast_df["Lower_Bound ($)"] <= forecast_df["Forecast ($)"]).all()
    assert (forecast_df["Upper_Bound ($)"] >= forecast_df["Forecast ($)"]).all()


def test_recursive_feature_state_updates():
    history = [100.0, 120.0, 110.0, 130.0]
    next_date = pd.to_datetime("2023-01-06")
    feats = update_recursive_features(history, next_date, is_holiday=0)
    assert feats["Lag_1"] == 130.0
    assert feats["Lag_2"] == 110.0
    assert feats["Week_of_Year"] == 1


# ---------------------------------------------------------
# 6. INVENTORY REPLENISHMENT LOGIC TESTS
# ---------------------------------------------------------

def test_inventory_policy_calculations_consistent_units():
    # Monetary Planning Scenario
    monetary_res = calculate_inventory_policy(
        forecast_weekly=20000.0,
        demand_std=2000.0,
        lead_time_weeks=2,
        review_period_weeks=1,
        service_level_pct=95.0,
        current_on_hand=15000.0,
        incoming_orders=5000.0,
        backorders=0.0,
        is_monetary_scenario=True
    )
    assert monetary_res["lead_time_demand"] == 40000.0
    assert monetary_res["safety_stock"] > 0
    assert monetary_res["reorder_point"] > monetary_res["lead_time_demand"]
    assert monetary_res["order_quantity"] >= 0
    assert monetary_res["unit_label"] == "$"

    # Demonstration Unit Scenario
    unit_res = calculate_inventory_policy(
        forecast_weekly=500.0,
        demand_std=50.0,
        lead_time_weeks=2,
        review_period_weeks=1,
        service_level_pct=95.0,
        current_on_hand=600.0,
        incoming_orders=0.0,
        backorders=0.0,
        unit_cost=15.0,
        is_monetary_scenario=False
    )
    assert unit_res["lead_time_demand"] == 1000.0
    assert unit_res["unit_label"] == "units"
    assert "DEMONSTRATION SCENARIO" in unit_res["disclaimer"]


# ---------------------------------------------------------
# 7. GRACEFUL HANDLING OF SHORT SERIES / EDGE CASES
# ---------------------------------------------------------

def test_graceful_handling_short_series():
    # Synthetic series of only 25 weeks (insufficient for 52-week seasonal lag)
    dates = pd.date_range("2021-01-01", periods=25, freq="W-FRI")
    short_df = pd.DataFrame({
        "Store": 99,
        "Dept": 99,
        "Date": dates,
        "Weekly_Sales": np.random.uniform(5000, 10000, size=25),
        "IsHoliday": [0] * 25
    })
    cleaned = preprocess_series(short_df)
    feat_df = build_feature_dataframe(cleaned, include_lag_52=False)
    
    # Lag_52 should not be generated for a 25-week series
    assert "Lag_52" not in feat_df.columns

    # Naive model still works
    naive = NaiveForecaster()
    naive.fit(short_df["Weekly_Sales"])
    pred = naive.predict(steps=4)
    assert len(pred) == 4


def test_holiday_shock_evaluation(merged_dataset):
    s, _ = get_series(merged_dataset, 1, 1)
    clean = preprocess_series(s)
    f = build_feature_dataframe(clean)
    board, preds, test_df = evaluate_holdout(f, train_ratio=0.8)
    
    # Test analyze_residuals_and_holidays
    res_analysis = analyze_residuals_and_holidays(test_df, preds["naive"])
    assert "holiday_stats" in res_analysis
    assert "Holiday" in res_analysis["holiday_stats"] or "Non-Holiday" in res_analysis["holiday_stats"]
    
    # Test evaluate_holdout_shocks_all_models
    shock_df = evaluate_holdout_shocks_all_models(test_df, preds)
    assert len(shock_df) == len(preds)
    assert "Holiday Shock WAPE (%)" in shock_df.columns
    assert "Non-Holiday WAPE (%)" in shock_df.columns
    assert "Holiday Weeks (N)" in shock_df.columns

