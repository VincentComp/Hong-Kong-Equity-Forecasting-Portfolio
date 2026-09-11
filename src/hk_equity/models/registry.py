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

from typing import Any

import pandas as pd

from src.hk_equity.models.baseline import (
    ewma_forecast,
    moving_average_forecast,
    zero_return_forecast,
)
from src.hk_equity.models.general_regression import (
    get_regression_backtest_forecasts,
    get_regression_forecast,
)


def predict_zero(
    weekly_returns: pd.DataFrame,
    parameters: dict[str, Any],
) -> pd.Series:
    """Predict a 0% next-week return for every stock."""

    return pd.Series(
        data=0.0,
        index=weekly_returns.columns,
        dtype=float,
        name="predicted_weekly_return",
    )


def predict_moving_average(
    weekly_returns: pd.DataFrame,
    parameters: dict[str, Any],
) -> pd.Series:
    """Predict next-week returns from the recent simple average return."""

    lookback_weeks = int(
        parameters["lookback_weeks"]
    )

    if lookback_weeks <= 0:
        raise ValueError(
            "lookback_weeks must be greater than zero."
        )

    if len(weekly_returns) < lookback_weeks:
        raise ValueError(
            f"Moving Average needs at least {lookback_weeks} "
            "completed weekly returns."
        )

    return pd.Series(
        weekly_returns.tail(lookback_weeks).mean(),
        name="predicted_weekly_return",
    )


def predict_ewma(
    weekly_returns: pd.DataFrame,
    parameters: dict[str, Any],
) -> pd.Series:
    """Predict next-week returns from an exponentially weighted average."""

    span_weeks = int(
        parameters["span_weeks"]
    )

    if span_weeks <= 0:
        raise ValueError(
            "span_weeks must be greater than zero."
        )

    if len(weekly_returns) < span_weeks:
        raise ValueError(
            f"EWMA needs at least {span_weeks} completed weekly returns."
        )

    return pd.Series(
        weekly_returns
        .ewm(span=span_weeks, adjust=False)
        .mean()
        .iloc[-1],
        name="predicted_weekly_return",
    )
    #a = 2 / (1+span)
    #EWMA(t) = a*R(t) + (1-a)*EWMA(t-1)


def _get_model_name(
    model_config: dict[str, Any],
) -> str:
    """Read and normalize the selected model name."""

    try:
        model_name = model_config["model"]["name"]
    except KeyError as error:
        raise KeyError(
            "Model config must contain model.name."
        ) from error

    return str(model_name).lower()


def _get_available_model_names(
    registry: dict[str, object],
) -> str:
    """Return sorted model names for clear error messages."""

    return ", ".join(sorted(registry.keys()))


def _get_regression_settings(
    model_config: dict[str, Any],
) -> dict[str, Any]:
    """Merge model parameters with regression feature settings.

    Baseline models continue using only ``parameters``. Regression additionally
    needs model-specific feature settings such as selected_columns and the
    processed feature-map path.
    """

    parameters = model_config.get(
        "parameters",
        {},
    )

    feature_settings = model_config.get(
        "features",
        {},
    )

    return {
        **parameters,
        **feature_settings,
    }


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
    model_config: dict[str, Any],
    benchmark_returns: pd.Series | None = None,
) -> pd.Series:
    """Generate one next-week predicted return for every stock.

    Baseline models use only portfolio weekly returns. The general regression
    model additionally uses benchmark returns, normally HSI weekly returns.
    """

    model_name = _get_model_name(model_config)

    if model_name == "general_regression":
        if benchmark_returns is None:
            raise ValueError(
                "benchmark_returns is required for general_regression."
            )

        regression_settings = _get_regression_settings(
            model_config=model_config,
        )

        return get_regression_forecast(
            weekly_returns=weekly_returns,
            benchmark_returns=benchmark_returns,
            parameters=regression_settings,
        )

    parameters = model_config.get(
        "parameters",
        {},
    )

    if model_name not in LIVE_MODEL_REGISTRY:
        available_models = _get_available_model_names(
            {
                **LIVE_MODEL_REGISTRY,
                "general_regression": get_regression_forecast,
            }
        )

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
    model_config: dict[str, Any],
    benchmark_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Generate historical one-step-ahead forecasts for every stock and week.

    Baseline backtests use the existing shifted DataFrame functions. General
    regression uses an expanding-window walk-forward procedure so each target
    week is predicted only from earlier feature-map observations.
    """

    model_name = _get_model_name(model_config)

    if model_name == "general_regression":
        if benchmark_returns is None:
            raise ValueError(
                "benchmark_returns is required for general_regression."
            )

        regression_settings = _get_regression_settings(
            model_config=model_config,
        )

        return get_regression_backtest_forecasts(
            weekly_returns=weekly_returns,
            parameters=regression_settings,
        )

    parameters = model_config.get(
        "parameters",
        {},
    )

    if model_name not in BACKTEST_MODEL_REGISTRY:
        available_models = _get_available_model_names(
            {
                **BACKTEST_MODEL_REGISTRY,
                "general_regression": get_regression_backtest_forecasts,
            }
        )

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