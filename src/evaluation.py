"""
evaluation.py
Robust evaluation module supporting:
- Fair metric computations (MAE, RMSE, MAPE, WAPE with zero-division handling)
- Chronological train-test holdout
- Rolling-origin expanding-window cross validation
- Programmatic model selection and leaderboard generation
- Residual and holiday demand shock diagnostics
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.forecasting_models import BaseForecaster, get_all_models


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error in dollars."""
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error in dollars."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
    """
    Mean Absolute Percentage Error (%).
    Handles zero actual values safely using epsilon.
    """
    denominator = np.abs(y_true)
    # If any zeros are present, use safe denominator
    denominator = np.where(denominator < epsilon, epsilon, denominator)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)


def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
    """
    Weighted Absolute Percentage Error (%).
    WAPE = (sum(|y - y_hat|) / sum(|y|)) * 100
    Highly recommended for retail because it weights errors by sales volume
    and avoids distortions from near-zero sales weeks.
    """
    denom = np.sum(np.abs(y_true))
    if denom < epsilon:
        return 0.0
    return float((np.sum(np.abs(y_true - y_pred)) / denom) * 100.0)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes MAE, RMSE, MAPE, and WAPE in a single dictionary."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    return {
        "MAE": round(calculate_mae(y_t, y_p), 2),
        "RMSE": round(calculate_rmse(y_t, y_p), 2),
        "MAPE": round(calculate_mape(y_t, y_p), 2),
        "WAPE": round(calculate_wape(y_t, y_p), 2)
    }


def evaluate_holdout(
    df_features: pd.DataFrame,
    target_col: str = "Weekly_Sales",
    train_ratio: float = 0.8,
    models: Optional[Dict[str, BaseForecaster]] = None
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], pd.DataFrame]:
    """
    Performs strict chronological holdout evaluation across all models.
    Preserves 80:20 chronological split without shuffling.
    
    Returns:
    - leaderboard_df: Summary metrics per model
    - predictions_dict: Dictionary mapping model_key -> array of test predictions
    - test_data_df: The test subset with actuals and dates
    """
    models = models or get_all_models()
    
    # Drop rows that have NaN in lag/rolling warmup features (e.g. initial 52 weeks)
    warmup_cols = [c for c in ["Lag_1", "Lag_2", "Lag_3", "Lag_4", "Lag_52", "Rolling_Mean_13"] if c in df_features.columns]
    clean_df = df_features.dropna(subset=warmup_cols).reset_index(drop=True)
    
    n = len(clean_df)
    train_size = int(n * train_ratio)
    
    train_df = clean_df.iloc[:train_size].reset_index(drop=True)
    test_df = clean_df.iloc[train_size:].reset_index(drop=True)
    
    y_train = train_df[target_col]
    y_test = test_df[target_col].values
    n_test = len(test_df)
    
    results = []
    predictions = {}
    
    for key, model in models.items():
        try:
            # Fit model
            if model.model_type in ["ml", "baseline"] and hasattr(model, "feature_names") and model.feature_names:
                model.fit(y=y_train, X=train_df)
                # One-step rolling predictions on test set
                preds = model.predict(steps=n_test, X=test_df)
            else:
                # Time-series statistical models (fit on y_train, predict horizon)
                model.fit(y=y_train)
                preds = model.predict(steps=n_test)
            
            # Clip negative predictions to 0 for retail sales realism
            preds = np.clip(np.asarray(preds, dtype=float), a_min=0.0, a_max=None)
            predictions[key] = preds
            
            metrics = compute_all_metrics(y_test, preds)
            results.append({
                "Model Key": key,
                "Model": model.name,
                "Type": model.model_type.capitalize(),
                "Evaluation Method": "Chronological Holdout (80/20)",
                "Successful Folds": 1,
                "Evaluated Obs": n_test,
                "MAPE (%)": metrics["MAPE"],
                "MAE ($)": metrics["MAE"],
                "RMSE ($)": metrics["RMSE"],
                "WAPE (%)": metrics["WAPE"],
                "Status": "Valid"
            })
        except Exception as e:
            results.append({
                "Model Key": key,
                "Model": model.name,
                "Type": model.model_type.capitalize(),
                "Evaluation Method": "Chronological Holdout (80/20)",
                "Successful Folds": 0,
                "Evaluated Obs": 0,
                "MAPE (%)": np.nan,
                "MAE ($)": np.nan,
                "RMSE ($)": np.nan,
                "WAPE (%)": np.nan,
                "Status": f"Failed: {str(e)[:40]}"
            })
            
    leaderboard = pd.DataFrame(results)
    return leaderboard, predictions, test_df


def evaluate_rolling_origin(
    df_features: pd.DataFrame,
    target_col: str = "Weekly_Sales",
    horizon: int = 4,
    n_splits: int = 4,
    models: Optional[Dict[str, BaseForecaster]] = None
) -> Tuple[pd.DataFrame, Dict[str, List[np.ndarray]]]:
    """
    Performs expanding-window rolling-origin validation across multiple forecast origins.
    At each origin:
    - Train on all available past data
    - Forecast the next `horizon` weeks
    - Compare against actual observations
    - Average the performance metrics across origins
    """
    models = models or get_all_models()
    warmup_cols = [c for c in ["Lag_1", "Lag_2", "Lag_3", "Lag_4", "Lag_52", "Rolling_Mean_13"] if c in df_features.columns]
    clean_df = df_features.dropna(subset=warmup_cols).reset_index(drop=True)
    n = len(clean_df)
    
    # Calculate starting origin
    total_test_span = horizon * n_splits
    if n - total_test_span < 20:
        # If dataset is tight, adjust splits to fit available history
        n_splits = max(2, (n - 20) // horizon)
        total_test_span = horizon * n_splits

    origin_indices = [n - total_test_span + (i * horizon) for i in range(n_splits)]
    
    model_fold_metrics = {k: {"MAE": [], "RMSE": [], "MAPE": [], "WAPE": []} for k in models.keys()}
    model_predictions = {k: [] for k in models.keys()}
    model_errors = {k: None for k in models.keys()}
    
    for origin in origin_indices:
        train_df = clean_df.iloc[:origin].reset_index(drop=True)
        test_df = clean_df.iloc[origin : origin + horizon].reset_index(drop=True)
        
        y_train = train_df[target_col]
        y_test = test_df[target_col].values
        
        fresh_forecasters = get_all_models()
        for key in models.keys():
            m = fresh_forecasters[key]
            try:
                if m.model_type in ["ml", "baseline"] and hasattr(m, "feature_names") and m.feature_names:
                    m.fit(y=y_train, X=train_df)
                    preds = m.predict(steps=horizon, X=test_df)
                else:
                    m.fit(y=y_train)
                    preds = m.predict(steps=horizon)
                    
                preds = np.clip(np.asarray(preds, dtype=float), a_min=0.0, a_max=None)
                model_predictions[key].append(preds)
                
                fold_m = compute_all_metrics(y_test, preds)
                for metric_name in ["MAE", "RMSE", "MAPE", "WAPE"]:
                    model_fold_metrics[key][metric_name].append(fold_m[metric_name])
            except Exception as e:
                model_errors[key] = str(e)
                
    summary = []
    for key, model in models.items():
        metrics_dict = model_fold_metrics[key]
        succ_folds = len(metrics_dict["MAE"])
        if succ_folds > 0:
            summary.append({
                "Model Key": key,
                "Model": model.name,
                "Type": model.model_type.capitalize(),
                "Evaluation Method": f"Rolling-Origin CV ({succ_folds}/{n_splits} folds x {horizon} wks)",
                "Successful Folds": succ_folds,
                "Evaluated Obs": succ_folds * horizon,
                "MAPE (%)": round(float(np.mean(metrics_dict["MAPE"])), 2),
                "MAE ($)": round(float(np.mean(metrics_dict["MAE"])), 2),
                "RMSE ($)": round(float(np.mean(metrics_dict["RMSE"])), 2),
                "WAPE (%)": round(float(np.mean(metrics_dict["WAPE"])), 2),
                "Status": "Valid" if succ_folds == n_splits else f"Partial ({succ_folds}/{n_splits} folds)"
            })
        else:
            summary.append({
                "Model Key": key,
                "Model": model.name,
                "Type": model.model_type.capitalize(),
                "Evaluation Method": f"Rolling-Origin CV (0/{n_splits} folds x {horizon} wks)",
                "Successful Folds": 0,
                "Evaluated Obs": 0,
                "MAPE (%)": np.nan,
                "MAE ($)": np.nan,
                "RMSE ($)": np.nan,
                "WAPE (%)": np.nan,
                "Status": f"Failed: {model_errors[key]}"
            })
            
    leaderboard = pd.DataFrame(summary)
    return leaderboard, model_predictions


def build_ranked_leaderboard(
    leaderboard_df: pd.DataFrame,
    primary_metric: str = "WAPE (%)",
    ascending: bool = True
) -> Tuple[pd.DataFrame, str]:
    """
    Ranks valid models by the chosen primary metric.
    Returns ranked DataFrame and the key of the winning model programmatically.
    """
    ranked = leaderboard_df.copy()
    valid_mask = ranked["Status"] == "Valid"
    
    valid_df = ranked[valid_mask].sort_values(by=primary_metric, ascending=ascending).reset_index(drop=True)
    valid_df["Rank"] = valid_df.index + 1
    
    invalid_df = ranked[~valid_mask].copy()
    invalid_df["Rank"] = np.nan
    
    final_df = pd.concat([valid_df, invalid_df], ignore_index=True)
    
    winning_model_key = valid_df.iloc[0]["Model Key"] if len(valid_df) > 0 else "naive"
    return final_df, winning_model_key


def analyze_residuals_and_holidays(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    target_col: str = "Weekly_Sales"
) -> Dict[str, Any]:
    """
    Computes holiday vs non-holiday forecast performance and identifies peak error periods.
    """
    analysis = test_df[["Date", target_col]].copy()
    if "IsHoliday" in test_df.columns:
        analysis["IsHoliday"] = test_df["IsHoliday"]
    else:
        analysis["IsHoliday"] = 0

    analysis["Actual"] = test_df[target_col].values
    analysis["Predicted"] = predictions
    analysis["Error"] = analysis["Actual"] - analysis["Predicted"]
    analysis["Abs_Error"] = np.abs(analysis["Error"])
    analysis["Pct_Error"] = (analysis["Abs_Error"] / np.clip(analysis["Actual"], 1.0, None)) * 100.0

    # Holiday performance breakdown
    holiday_stats = {}
    for is_hol in [0, 1]:
        subset = analysis[analysis["IsHoliday"] == is_hol]
        label = "Holiday" if is_hol == 1 else "Non-Holiday"
        if len(subset) > 0:
            holiday_stats[label] = {
                "Count": len(subset),
                "Mean_Actual": round(float(subset["Actual"].mean()), 2),
                "Mean_Predicted": round(float(subset["Predicted"].mean()), 2),
                "MAE": round(float(subset["Abs_Error"].mean()), 2),
                "RMSE": round(calculate_rmse(subset["Actual"].values, subset["Predicted"].values), 2),
                "MAPE": round(calculate_mape(subset["Actual"].values, subset["Predicted"].values), 2),
                "WAPE": round(calculate_wape(subset["Actual"].values, subset["Predicted"].values), 2)
            }

    # Top 5 error periods
    top_errors = analysis.sort_values(by="Abs_Error", ascending=False).head(5)[
        ["Date", "Actual", "Predicted", "Error", "Abs_Error", "IsHoliday"]
    ].to_dict(orient="records")

    return {
        "analysis_table": analysis,
        "holiday_stats": holiday_stats,
        "top_errors": top_errors,
        "residual_mean": float(analysis["Error"].mean()),
        "residual_std": float(analysis["Error"].std())
    }


def evaluate_holdout_shocks_all_models(
    test_df: pd.DataFrame,
    predictions_dict: Dict[str, np.ndarray],
    models_dict: Optional[Dict[str, BaseForecaster]] = None,
    target_col: str = "Weekly_Sales"
) -> pd.DataFrame:
    """
    Evaluates forecasting errors (WAPE and MAE) separately for holiday shock weeks
    and non-holiday baseline weeks across all candidate models in the holdout test set.
    Includes explicit sample counts (N) for rigorous sample-size transparency.
    """
    models_dict = models_dict or get_all_models()
    is_hol = test_df["IsHoliday"].values == 1 if "IsHoliday" in test_df.columns else np.zeros(len(test_df), dtype=bool)
    n_hol = int(np.sum(is_hol))
    n_non_hol = int(np.sum(~is_hol))
    y_true = test_df[target_col].values

    rows = []
    for key, preds in predictions_dict.items():
        model_name = models_dict[key].name if key in models_dict else key
        overall_wape = calculate_wape(y_true, preds)
        overall_mae = calculate_mae(y_true, preds)

        if n_hol > 0:
            hol_wape = calculate_wape(y_true[is_hol], preds[is_hol])
            hol_mae = calculate_mae(y_true[is_hol], preds[is_hol])
        else:
            hol_wape, hol_mae = np.nan, np.nan

        if n_non_hol > 0:
            non_wape = calculate_wape(y_true[~is_hol], preds[~is_hol])
            non_mae = calculate_mae(y_true[~is_hol], preds[~is_hol])
        else:
            non_wape, non_mae = np.nan, np.nan

        rows.append({
            "Model": model_name,
            "Overall WAPE (%)": round(overall_wape, 2),
            "Non-Holiday WAPE (%)": round(non_wape, 2),
            "Holiday Shock WAPE (%)": round(hol_wape, 2),
            "Non-Holiday MAE ($)": round(non_mae, 2),
            "Holiday Shock MAE ($)": round(hol_mae, 2),
            "Holiday Weeks (N)": n_hol,
            "Non-Holiday Weeks (N)": n_non_hol,
        })

    return pd.DataFrame(rows).sort_values("Overall WAPE (%)").reset_index(drop=True)

