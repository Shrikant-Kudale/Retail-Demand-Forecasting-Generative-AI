"""
forecasting_models.py
Implementation of 8 retail demand forecasting models with a consistent interface:
1. Naive Forecast (t-1)
2. Seasonal Naive Forecast (t-52)
3. Holt-Winters Exponential Smoothing
4. SARIMA Time-Series Model
5. Lag-based Linear Regression
6. Random Forest Regressor
7. Gradient Boosting Regressor
8. Improved Seasonal Regression (Lag 52 + Cyclical Seasonality + Holiday)
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.feature_engineering import FEATURE_SETS, get_available_features


class BaseForecaster:
    """Abstract base forecaster interface."""
    def __init__(self, name: str, model_type: str, description: str):
        self.name = name
        self.model_type = model_type  # 'baseline', 'statistical', 'ml'
        self.description = description
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.last_train_y: Optional[pd.Series] = None

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "BaseForecaster":
        raise NotImplementedError

    def predict(self, steps: int = 1, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        raise NotImplementedError


class NaiveForecaster(BaseForecaster):
    """
    Naive Forecast: Predicts future sales as the most recently observed weekly sales value (y_t = y_{t-1}).
    Serves as the foundational baseline to ensure complex models provide actual value.
    """
    def __init__(self):
        super().__init__(
            name="Naive (Last Value)",
            model_type="baseline",
            description="Predicts the last observed weekly sales value for all future periods."
        )
        self.last_value: float = 0.0

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "NaiveForecaster":
        if len(y) == 0:
            raise ValueError("Training series is empty.")
        self.last_value = float(y.iloc[-1])
        self.last_train_y = y
        self.is_fitted = True
        return self

    def predict(self, steps: int = 1, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        if X is not None and "Lag_1" in X.columns:
            # One-step ahead rolling prediction if Lag_1 is provided
            return X["Lag_1"].values
        return np.full(steps, self.last_value, dtype=float)


class SeasonalNaiveForecaster(BaseForecaster):
    """
    Seasonal Naive: Predicts sales equal to the sales observed in the same week last year (y_t = y_{t-52}).
    Captures strong 52-week annual retail seasonality without statistical parameters.
    """
    def __init__(self, seasonal_period: int = 52):
        super().__init__(
            name="Seasonal Naive (t-52)",
            model_type="baseline",
            description=f"Predicts sales from {seasonal_period} weeks earlier (same week last year)."
        )
        self.seasonal_period = seasonal_period
        self.history: Optional[np.ndarray] = None

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "SeasonalNaiveForecaster":
        if len(y) < self.seasonal_period:
            # Fallback if series has less than 52 weeks
            self.history = y.values
            self.last_train_y = y
            self.is_fitted = True
            return self
        self.history = y.values
        self.last_train_y = y
        self.is_fitted = True
        return self

    def predict(self, steps: int = 1, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        if X is not None and "Lag_52" in X.columns and not X["Lag_52"].isnull().any():
            return X["Lag_52"].values
        
        preds = []
        hist_len = len(self.history)
        for step in range(steps):
            idx = hist_len - self.seasonal_period + (step % self.seasonal_period)
            if idx >= 0 and idx < hist_len:
                preds.append(self.history[idx])
            else:
                preds.append(self.history[-1])
        return np.array(preds, dtype=float)


class HoltWintersForecaster(BaseForecaster):
    """
    Holt-Winters Exponential Smoothing:
    Decomposes demand into level, additive trend, and additive 52-week seasonal patterns.
    """
    def __init__(self, seasonal_periods: int = 52):
        super().__init__(
            name="Holt-Winters Exponential Smoothing",
            model_type="statistical",
            description="Statistical decomposition capturing level, linear trend, and 52-week seasonal cycles."
        )
        self.seasonal_periods = seasonal_periods
        self.fitted_model: Optional[Any] = None

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "HoltWintersForecaster":
        # Ensure a clean 0-indexed RangeIndex series for statsmodels compatibility
        y_clean = pd.Series(y.values, dtype=float, index=pd.RangeIndex(len(y)))
        self.last_train_y = y_clean
        n = len(y_clean)

        try:
            if n >= 2 * self.seasonal_periods:
                model = ExponentialSmoothing(
                    y_clean,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=self.seasonal_periods,
                    initialization_method="estimated"
                )
                self.fitted_model = model.fit(optimized=True)
            elif n >= 20:
                # When n < 104, 52 seasonal parameters cannot be reliably estimated.
                # Use Holt's additive linear trend model.
                model = ExponentialSmoothing(
                    y_clean,
                    trend="add",
                    seasonal=None,
                    initialization_method="estimated"
                )
                self.fitted_model = model.fit(optimized=True)
            else:
                # Fallback to simple exponential smoothing for very short series
                model = ExponentialSmoothing(y_clean, trend=None, seasonal=None)
                self.fitted_model = model.fit(optimized=True)
            self.is_fitted = True
        except Exception:
            # Robust fallback to simple exponential smoothing
            model = ExponentialSmoothing(y_clean, trend=None, seasonal=None)
            self.fitted_model = model.fit(optimized=True)
            self.is_fitted = True

        return self

    def predict(self, steps: int = 1, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self.is_fitted or self.fitted_model is None:
            raise RuntimeError("Model is not fitted.")
        forecast = self.fitted_model.forecast(steps)
        return np.asarray(forecast, dtype=float)


class SARIMAForecaster(BaseForecaster):
    """
    SARIMA (Seasonal Autoregressive Integrated Moving Average):
    Rigorous statistical time-series model modeling autocorrelation, differencing for stationarity,
    and annual seasonal moving average components.
    """
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 0), seasonal_order: Tuple[int, int, int, int] = (1, 0, 0, 52)):
        super().__init__(
            name="SARIMA",
            model_type="statistical",
            description="Statistical state-space model modeling autoregression, differencing, and seasonal lag correlation."
        )
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted_model: Optional[Any] = None

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "SARIMAForecaster":
        # Ensure a clean 0-indexed RangeIndex series for statsmodels compatibility
        y_clean = pd.Series(y.values, dtype=float, index=pd.RangeIndex(len(y)))
        self.last_train_y = y_clean
        n = len(y_clean)

        try:
            if n >= 2 * 52:
                model = SARIMAX(
                    y_clean,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                self.fitted_model = model.fit(disp=False, maxiter=50)
            else:
                # When n < 104, seasonal order of period 52 is overparameterized.
                # Fit non-seasonal ARIMA(1, 1, 0) for rapid, stable convergence.
                model = SARIMAX(
                    y_clean,
                    order=self.order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                self.fitted_model = model.fit(disp=False, maxiter=35)
            self.is_fitted = True
        except Exception:
            # Fallback simpler ARIMA(1,0,0)
            model = SARIMAX(y_clean, order=(1, 0, 0), enforce_stationarity=False, enforce_invertibility=False)
            self.fitted_model = model.fit(disp=False, maxiter=25)
            self.is_fitted = True

        return self

    def predict(self, steps: int = 1, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self.is_fitted or self.fitted_model is None:
            raise RuntimeError("Model is not fitted.")
        forecast = self.fitted_model.forecast(steps)
        return np.asarray(forecast, dtype=float)


class LagLinearRegressionForecaster(BaseForecaster):
    """
    Lag-based Linear Regression:
    Replicates the existing baseline model from the original notebook using Lags 1, 2, 3, and 4.
    """
    def __init__(self):
        super().__init__(
            name="Lag Linear Regression (Baseline)",
            model_type="baseline",
            description="Linear regression predicting sales from recent weekly lags (1 to 4 weeks)."
        )
        self.model = LinearRegression()
        self.feature_names = FEATURE_SETS["lag_only"]

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "LagLinearRegressionForecaster":
        if X is None:
            raise ValueError("Feature DataFrame X is required for Lag Linear Regression.")
        
        cols = get_available_features(X, self.feature_names)
        if not cols:
            raise ValueError("Required lag features (Lag_1 to Lag_4) not found in X.")
        self.feature_names = cols
        
        self.model.fit(X[cols], y)
        self.last_train_y = y
        self.is_fitted = True
        return self

    def predict(self, steps: int = 1, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        if X is None:
            raise ValueError("Feature DataFrame X is required for prediction.")
        return self.model.predict(X[self.feature_names])


class ImprovedSeasonalRegressionForecaster(BaseForecaster):
    """
    Improved Seasonal Regression:
    Replicates the improved model from the original notebook, incorporating
    recent lags (1-4), 52-week seasonal lag, cyclical sinusoidal calendar features, and holiday flags.
    """
    def __init__(self):
        super().__init__(
            name="Improved Seasonal Regression",
            model_type="ml",
            description="Linear model combining 4-week lags, 52-week annual lag, cyclical week harmonics, and holiday effects."
        )
        self.model = LinearRegression()
        self.feature_names = FEATURE_SETS["seasonal"]

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "ImprovedSeasonalRegressionForecaster":
        if X is None:
            raise ValueError("Feature DataFrame X is required.")
        cols = get_available_features(X, self.feature_names)
        self.feature_names = cols
        self.model.fit(X[cols], y)
        self.last_train_y = y
        self.is_fitted = True
        return self

    def predict(self, steps: int = 1, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        if X is None:
            raise ValueError("Feature DataFrame X is required.")
        return self.model.predict(X[self.feature_names])


class RandomForestForecaster(BaseForecaster):
    """
    Random Forest Regressor:
    Ensemble of decision trees capturing nonlinear interactions between recent lags,
    rolling statistics, calendar cyclicality, and store promotions.
    """
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, random_state: int = 42):
        super().__init__(
            name="Random Forest",
            model_type="ml",
            description="Tree ensemble modeling non-linear interactions across lags, rolling statistics, and external features."
        )
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=1  # single-threaded: safe on serverless runtimes
        )
        self.feature_names = FEATURE_SETS["tree_all"]

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "RandomForestForecaster":
        if X is None:
            raise ValueError("Feature DataFrame X is required.")
        cols = get_available_features(X, self.feature_names)
        self.feature_names = cols
        self.model.fit(X[cols], y)
        self.last_train_y = y
        self.is_fitted = True
        return self

    def predict(self, steps: int = 1, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        if X is None:
            raise ValueError("Feature DataFrame X is required.")
        return self.model.predict(X[self.feature_names])


class GradientBoostingForecaster(BaseForecaster):
    """
    Gradient Boosting Regressor:
    Sequentially builds trees that minimize forecasting residuals, with strong ability
    to model promotional demand spikes and macroeconomic conditions.
    """
    def __init__(self, n_estimators: int = 100, max_depth: int = 4, learning_rate: float = 0.08, random_state: int = 42):
        super().__init__(
            name="Gradient Boosting",
            model_type="ml",
            description="Sequentially optimized boosted trees tailored for promotional shocks and seasonal surges."
        )
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state
        )
        self.feature_names = FEATURE_SETS["tree_all"]

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "GradientBoostingForecaster":
        if X is None:
            raise ValueError("Feature DataFrame X is required.")
        cols = get_available_features(X, self.feature_names)
        self.feature_names = cols
        self.model.fit(X[cols], y)
        self.last_train_y = y
        self.is_fitted = True
        return self

    def predict(self, steps: int = 1, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        if X is None:
            raise ValueError("Feature DataFrame X is required.")
        return self.model.predict(X[self.feature_names])


def get_all_models() -> Dict[str, BaseForecaster]:
    """
    Returns an instantiated dictionary of all 8 candidate forecasting models.
    """
    return {
        "naive": NaiveForecaster(),
        "seasonal_naive": SeasonalNaiveForecaster(seasonal_period=52),
        "holt_winters": HoltWintersForecaster(seasonal_periods=52),
        "sarima": SARIMAForecaster(),
        "lag_linear_regression": LagLinearRegressionForecaster(),
        "improved_seasonal_regression": ImprovedSeasonalRegressionForecaster(),
        "random_forest": RandomForestForecaster(),
        "gradient_boosting": GradientBoostingForecaster(),
    }
