"""
visualization.py
Reusable Plotly charts and publication-quality Matplotlib helpers
for notebooks and offline reporting (the web app renders with Plotly.js).
Styling follows the Premium Neutral Analytics design system:
Warm neutral surfaces (#FFFFFF), subtle borders (#E6E0D6), forest slate (#34443D),
muted sage (#5B7566), warm amber (#C07D3E), and restrained crimson (#B84A39).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None


# Premium Neutral Analytics Palette
PALETTE = {
    "actual": "#34443D",       # Deep forest slate
    "forecast": "#C07D3E",     # Warm amber / cognac
    "rolling": "#5B7566",      # Muted sage
    "baseline": "#9CA39E",     # Neutral slate gray
    "holiday": "#B84A39",      # Restrained brick crimson
    "interval": "rgba(192, 125, 62, 0.16)",  # Shaded 95% prediction interval band
    "card_bg": "#FFFFFF",
    "grid": "#EFEBE3",
    "border": "#E6E0D6",
    "text": "#292D2B",
    "subtext": "#747872",
}

FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def apply_chart_theme(
    fig: go.Figure,
    title: str,
    x_title: Optional[str] = None,
    y_title: Optional[str] = None,
    show_legend: bool = True
) -> go.Figure:
    """Applies a consistent Premium Neutral Analytics layout and typography to Plotly figures."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(family=FONT_FAMILY, size=15, color=PALETTE["text"]),
            x=0.01,
            xanchor="left",
            y=0.97,
            yanchor="top",
        ),
        paper_bgcolor=PALETTE["card_bg"],
        plot_bgcolor=PALETTE["card_bg"],
        font=dict(family=FONT_FAMILY, color=PALETTE["text"], size=12),
        margin=dict(l=55, r=25, t=65, b=45),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor=PALETTE["border"],
            font=dict(family=FONT_FAMILY, size=12, color=PALETTE["text"]),
        ),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(family=FONT_FAMILY, size=11, color=PALETTE["subtext"]),
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor=PALETTE["border"],
            borderwidth=0,
        ),
    )

    fig.update_xaxes(
        title=dict(text=x_title, font=dict(family=FONT_FAMILY, size=12, color=PALETTE["subtext"])) if x_title else None,
        tickfont=dict(family=FONT_FAMILY, size=11, color=PALETTE["subtext"]),
        showgrid=True,
        gridcolor=PALETTE["grid"],
        gridwidth=1,
        showline=True,
        linecolor=PALETTE["border"],
        linewidth=1,
        zeroline=False,
    )

    fig.update_yaxes(
        title=dict(text=y_title, font=dict(family=FONT_FAMILY, size=12, color=PALETTE["subtext"])) if y_title else None,
        tickfont=dict(family=FONT_FAMILY, size=11, color=PALETTE["subtext"]),
        showgrid=True,
        gridcolor=PALETTE["grid"],
        gridwidth=1,
        showline=True,
        linecolor=PALETTE["border"],
        linewidth=1,
        zeroline=False,
    )

    return fig


def plot_sales_trend_plotly(
    df: pd.DataFrame,
    title: str = "Weekly Sales Trend & Rolling Average"
) -> go.Figure:
    """Creates an interactive Plotly sales trend chart with rolling average."""
    fig = go.Figure()

    # Calculate 4-week rolling average backward using only current and previous observations (no lookahead)
    if "Rolling_Mean_4" in df.columns and not df["Rolling_Mean_4"].isnull().all():
        rolling_series = df["Rolling_Mean_4"]
    else:
        rolling_series = df["Weekly_Sales"].rolling(window=4, min_periods=1).mean()

    # 1. Actual weekly sales line
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Weekly_Sales"],
        mode="lines+markers",
        name="Weekly Sales ($)",
        line=dict(color=PALETTE["actual"], width=2.0),
        marker=dict(size=4.5, color=PALETTE["actual"], symbol="circle"),
        hovertemplate="<b>Weekly Sales:</b> $%{y:,.2f}<extra></extra>"
    ))

    # 2. 4-Week Rolling Average line (clearly distinguishable muted sage dash)
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=rolling_series,
        mode="lines",
        name="4-Week Rolling Avg",
        line=dict(color=PALETTE["rolling"], width=2.4, dash="dash"),
        hovertemplate="<b>4-Week Rolling Avg:</b> $%{y:,.2f}<extra></extra>"
    ))

    # 3. Highlight holiday weeks
    if "IsHoliday" in df.columns:
        holidays = df[df["IsHoliday"] == 1]
        fig.add_trace(go.Scatter(
            x=holidays["Date"],
            y=holidays["Weekly_Sales"],
            mode="markers",
            name="Holiday Weeks",
            marker=dict(color=PALETTE["holiday"], size=8, symbol="diamond", line=dict(color="#FFFFFF", width=1)),
            hovertemplate="<b>Holiday Demand:</b> $%{y:,.2f}<extra></extra>"
        ))

    apply_chart_theme(fig, title=title, x_title="Date", y_title="Weekly Sales (USD)")
    fig.update_yaxes(tickprefix="$", tickformat=",")
    return fig


def plot_actual_vs_predicted_plotly(
    dates: pd.Series,
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    selected_model: str,
    baseline_model: Optional[str] = None
) -> go.Figure:
    """Plots actual sales vs predicted sales for selected model and optional baseline."""
    fig = go.Figure()

    # Actual Sales
    fig.add_trace(go.Scatter(
        x=dates,
        y=y_true,
        mode="lines+markers",
        name="Actual Sales",
        line=dict(color=PALETTE["actual"], width=2.2),
        marker=dict(size=4.5, color=PALETTE["actual"])
    ))

    # Baseline Model if provided
    if baseline_model and baseline_model in predictions_dict:
        fig.add_trace(go.Scatter(
            x=dates,
            y=predictions_dict[baseline_model],
            mode="lines",
            name=f"Baseline: {baseline_model}",
            line=dict(color=PALETTE["baseline"], width=1.8, dash="dot")
        ))

    # Selected Model
    if selected_model in predictions_dict:
        fig.add_trace(go.Scatter(
            x=dates,
            y=predictions_dict[selected_model],
            mode="lines+markers",
            name=f"Selected: {selected_model}",
            line=dict(color=PALETTE["forecast"], width=2.4),
            marker=dict(size=5, color=PALETTE["forecast"])
        ))

    apply_chart_theme(
        fig,
        title=f"Holdout Actual vs. Predicted Demand: {selected_model}",
        x_title="Date",
        y_title="Weekly Sales (USD)"
    )
    fig.update_yaxes(tickprefix="$", tickformat=",")
    return fig


def plot_future_forecast_plotly(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    history_tail: int = 26
) -> go.Figure:
    """Plots recent historical sales connected seamlessly to the future forecast with confidence interval."""
    fig = go.Figure()

    recent_history = history_df.tail(history_tail).copy()

    # Historical line
    fig.add_trace(go.Scatter(
        x=recent_history["Date"],
        y=recent_history["Weekly_Sales"],
        mode="lines+markers",
        name="Historical Sales",
        line=dict(color=PALETTE["actual"], width=2.0),
        marker=dict(size=4.5, color=PALETTE["actual"])
    ))

    # Forecast upper bound
    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Upper_Bound ($)"],
        mode="lines",
        name="Upper Bound (95% PI)",
        line=dict(width=0),
        showlegend=False
    ))

    # Forecast lower bound (fills area between upper and lower)
    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Lower_Bound ($)"],
        mode="lines",
        name="95% Prediction Interval",
        fill="tonexty",
        fillcolor=PALETTE["interval"],
        line=dict(width=0)
    ))

    # Point Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Forecast ($)"],
        mode="lines+markers",
        name="Point Forecast",
        line=dict(color=PALETTE["forecast"], width=2.8, dash="dash"),
        marker=dict(size=6.5, symbol="circle", color=PALETTE["forecast"], line=dict(color="#FFFFFF", width=1))
    ))

    apply_chart_theme(
        fig,
        title="Multi-Step Future Demand Forecast with 95% Prediction Bounds",
        x_title="Date",
        y_title="Weekly Sales (USD)"
    )
    fig.update_yaxes(tickprefix="$", tickformat=",")
    return fig


def plot_leaderboard_bar_plotly(
    leaderboard_df: pd.DataFrame,
    metric_col: str = "WAPE (%)"
) -> go.Figure:
    """Creates a horizontal ranked bar chart of model evaluation performance with neutral palette."""
    valid_df = leaderboard_df[leaderboard_df["Status"] == "Valid"].copy()
    valid_df = valid_df.sort_values(by=metric_col, ascending=False)

    # Cohesive neutral sequential color ramp from muted sage to forest slate
    n = len(valid_df)
    color_scale = [
        [0.0, "#9CA39E"],
        [0.35, "#7A8C80"],
        [0.7, "#5B7566"],
        [1.0, "#34443D"]
    ]

    fig = px.bar(
        valid_df,
        x=metric_col,
        y="Model",
        orientation="h",
        color=metric_col,
        color_continuous_scale=color_scale,
        text=metric_col,
        title=f"Model Performance Comparison by {metric_col}"
    )

    apply_chart_theme(fig, title=f"Model Performance Comparison by {metric_col}", x_title=metric_col, y_title="", show_legend=False)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        textfont=dict(family=FONT_FAMILY, size=11, color=PALETTE["text"]),
        marker=dict(line=dict(color=PALETTE["border"], width=1))
    )
    return fig


def plot_residuals_plotly(analysis_table: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    """Generates residual error over time and residual distribution charts with neutral styling."""
    # Chart 1: Residuals over time
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(
        x=analysis_table["Date"],
        y=analysis_table["Error"],
        mode="lines+markers",
        name="Residual (Actual - Pred)",
        line=dict(color=PALETTE["holiday"], width=1.8),
        marker=dict(size=5.5, color=PALETTE["holiday"], line=dict(color="#FFFFFF", width=0.8))
    ))
    fig_time.add_hline(y=0, line_dash="dash", line_color=PALETTE["actual"], line_width=1.2)
    apply_chart_theme(
        fig_time,
        title="Forecast Residuals Over Time",
        x_title="Date",
        y_title="Residual Error ($)",
        show_legend=False
    )
    fig_time.update_yaxes(tickprefix="$", tickformat=",")

    # Chart 2: Residual distribution
    fig_dist = px.histogram(
        analysis_table,
        x="Error",
        nbins=20,
        marginal="box",
        title="Distribution of Forecast Errors",
        color_discrete_sequence=[PALETTE["rolling"]]
    )
    apply_chart_theme(
        fig_dist,
        title="Distribution of Forecast Errors",
        x_title="Error ($)",
        y_title="Frequency",
        show_legend=False
    )
    fig_dist.update_traces(marker=dict(line=dict(color=PALETTE["border"], width=1)))
    fig_dist.update_xaxes(tickprefix="$", tickformat=",")

    return fig_time, fig_dist
