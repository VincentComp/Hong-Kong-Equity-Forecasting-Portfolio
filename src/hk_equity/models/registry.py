"""Register and dispatch forecasting models for live forecasts and backtests.

This module is the central model-selection layer of the project. It maps the
model name in a YAML configuration file to the correct Python forecasting
function, so scripts do not need repeated model-specific ``if/elif`` logic.

Live forecasting and historical backtesting use separate registries because
their outputs differ:

- Live models return one Series: one predicted next-week return per stock.
- Backtest models return one DataFrame: one historical forecast per
  stock-week observation.

All functions within each registry follow a common interface:

    forecast_function(weekly_returns, parameters)

Each model YAML configuration should also contain ``model.label``. The label
is used in reports, forecast tables, CSV files, and plots, so adding a new
model does not require changing ``get_model_label()``.
"""

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
    """Predict a 0% next-week return for every stock."""

    return pd.Series(
        data=0.0,
        index=weekly_returns.columns,
        dtype=float,
    )


def predict_moving_average(
    weekly_returns: pd.DataFrame,
    parameters: dict,
) -> pd.Series:
    """Predict next-week returns from the recent simple average return."""

    lookback_weeks = parameters["lookback_weeks"]

    if len(weekly_returns) < lookback_weeks:
        raise ValueError(
            f"Moving Average needs at least {lookback_weeks} "
            "completed weekly returns."
        )

    return weekly_returns.tail(lookback_weeks).mean()


def predict_ewma(
    weekly_returns: pd.DataFrame,
    parameters: dict,
) -> pd.Series:
    """Predict next-week returns from an exponentially weighted average."""

    span_weeks = parameters["span_weeks"]

    if len(weekly_returns) < span_weeks:
        raise ValueError(
            f"EWMA needs at least {span_weeks} completed weekly returns."
        )

    return (
        weekly_returns
        .ewm(span=span_weeks, adjust=False)
        .mean()
        .iloc[-1]
    )
    #a = 2 / (1+span)
    #EWMA(t) = a*R(t) + (1-a)*EWMA(t-1)


#==================================================================================

#Use the predictied function in registry.py
#Return a series
LIVE_MODEL_REGISTRY = {
    "zero": predict_zero,
    "moving_average": predict_moving_average,
    "ewma": predict_ewma,
}

#Use the forcast function in {model}.py
#Return a dataframe
BACKTEST_MODEL_REGISTRY = {
    "zero": zero_return_forecast,
    "moving_average": moving_average_forecast,
    "ewma": ewma_forecast,
}

#===================================================================================


def get_model_forecast(
    weekly_returns: pd.DataFrame,
    model_config: dict,
) -> pd.Series:
    """Generate one next-week predicted return for every stock."""

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
    """Generate historical one-step-ahead forecasts for every stock and week."""

    model_name = model_config["model"]["name"]
    parameters = model_config.get("parameters", {})

    if model_name not in BACKTEST_MODEL_REGISTRY:
        available_models = ", ".join(BACKTEST_MODEL_REGISTRY.keys())

        raise ValueError(
            f"Unknown backtest model name: '{model_name}'. "
            f"Available models: {available_models}"
        )

    forecast_function = BACKTEST_MODEL_REGISTRY[model_name]

    return forecast_function(
        weekly_returns=weekly_returns,
        parameters=parameters,
    )


def get_model_label(
    model_config: dict,
) -> str:
    """Return the configured readable model label for outputs.

    The label is stored under ``model.label`` in each model YAML file.
    If it is missing, this function falls back to ``model.name`` so existing
    or incomplete configurations do not immediately break the workflow.

    Args:
        model_config: Selected model YAML configuration loaded as a dictionary.

    Returns:
        A readable model label for reports, CSV outputs, and plots.
    """

    model_settings = model_config["model"]

    return model_settings.get(
        "label",
        model_settings["name"],
    )