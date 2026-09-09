from __future__ import annotations

import pandas as pd

from src.hk_equity.models.baseline import (
    ewma_forecast,
    moving_average_forecast,
    zero_return_forecast,
)


def predict_zero(
    weekly_returns: pd.DataFrame,
    parameters: dict,
) -> pd.Series:
    """Predict 0% next-week return for every stock."""

    return pd.Series(
        data=0.0,
        index=weekly_returns.columns,
        dtype=float,
    )


def predict_moving_average(
    weekly_returns: pd.DataFrame,
    parameters: dict,
) -> pd.Series:
    """Predict next-week return using the recent simple average return."""

    lookback_weeks = parameters["lookback_weeks"]

    if len(weekly_returns) < lookback_weeks:
        raise ValueError(
            f"Moving Average needs at least {lookback_weeks} "
            "completed weekly returns."
        )

    # Use the latest completed weekly returns to forecast the next week.
    return weekly_returns.tail(lookback_weeks).mean()


def predict_ewma(
    weekly_returns: pd.DataFrame,
    parameters: dict,
) -> pd.Series:
    """Predict next-week return using an exponentially weighted average."""

    span_weeks = parameters["span_weeks"]

    if len(weekly_returns) < span_weeks:
        raise ValueError(
            f"EWMA needs at least {span_weeks} completed weekly returns."
        )

    # EWMA gives higher weight to the latest completed weekly returns.
    return (
        weekly_returns
        .ewm(span=span_weeks, adjust=False)
        .mean()
        .iloc[-1]
    )


# Live Forecast Registry:
# YAML model name -> Python function that produces one next-week forecast Series.
LIVE_MODEL_REGISTRY = {
    "zero": predict_zero,
    "moving_average": predict_moving_average,
    "ewma": predict_ewma,
}


# Historical Backtest Registry:
# YAML model name -> Python function that produces a historical forecast DataFrame.
BACKTEST_MODEL_REGISTRY = {
    "zero": zero_return_forecast,
    "moving_average": moving_average_forecast,
    "ewma": ewma_forecast,
}


def get_model_forecast(
    weekly_returns: pd.DataFrame,
    model_config: dict,
) -> pd.Series:
    """Generate one next-week predicted return for each stock.

    This function is used by:
        scripts/run_weekly_forecast.py

    Every live model must return:
        pd.Series
        - index: ticker symbols
        - values: predicted next-week returns
    """

    model_name = model_config["model"]["name"]
    parameters = model_config.get("parameters", {})

    if model_name not in LIVE_MODEL_REGISTRY:
        available_models = ", ".join(LIVE_MODEL_REGISTRY.keys())

        raise ValueError(
            f"Unknown live model name: '{model_name}'. "
            f"Available models: {available_models}"
        )

    forecast_function = LIVE_MODEL_REGISTRY[model_name]

    return forecast_function(
        weekly_returns=weekly_returns,
        parameters=parameters,
    )


def get_backtest_forecasts(
    weekly_returns: pd.DataFrame,
    model_config: dict,
) -> pd.DataFrame:
    """Generate historical one-step-ahead forecasts for every stock.

    This function is used by:
        scripts/run_backtest.py

    Every historical forecast must align with actual weekly returns:
        forecast at week t uses information available before week t.

    The Moving Average and EWMA functions in baseline.py already use
    .shift(1), so they avoid look-ahead bias.
    """

    model_name = model_config["model"]["name"]
    parameters = model_config.get("parameters", {})

    if model_name not in BACKTEST_MODEL_REGISTRY:
        available_models = ", ".join(BACKTEST_MODEL_REGISTRY.keys())

        raise ValueError(
            f"Unknown backtest model name: '{model_name}'. "
            f"Available models: {available_models}"
        )

    forecast_function = BACKTEST_MODEL_REGISTRY[model_name]

    if model_name == "zero":
        return forecast_function(
            weekly_returns=weekly_returns,
        )

    if model_name == "moving_average":
        return forecast_function(
            weekly_returns=weekly_returns,
            lookback_weeks=parameters["lookback_weeks"],
        )

    if model_name == "ewma":
        return forecast_function(
            weekly_returns=weekly_returns,
            span_weeks=parameters["span_weeks"],
        )

    raise ValueError(
        f"Backtest implementation is missing for model: {model_name}"
    )


def get_model_label(
    model_config: dict,
) -> str:
    """Return a readable model name for reports, CSV outputs, and plots."""

    model_name = model_config["model"]["name"]
    parameters = model_config.get("parameters", {})

    if model_name == "zero":
        return "Zero Return Baseline"

    if model_name == "moving_average":
        lookback = parameters["lookback_weeks"]
        return f"{lookback}-Week Moving Average"

    if model_name == "ewma":
        span = parameters["span_weeks"]
        return f"{span}-Week EWMA"

    return model_name