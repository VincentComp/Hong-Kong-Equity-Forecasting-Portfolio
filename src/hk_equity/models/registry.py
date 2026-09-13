"""Register and dispatch forecasting models for live forecasts and backtests.

This module is the central model-selection layer of the project. It maps the
model name in a YAML configuration file to a ModelAdapter.

The registry only dispatches model calls. Model-specific fitting, feature
engineering, sequence construction, and prediction logic belongs in the
corresponding model module.

Each model YAML configuration should contain ``model.name`` and normally
contains ``model.label``, ``model.version``, and ``model.run_id``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.hk_equity.models.adapters import (
    ModelAdapter,
)
from src.hk_equity.models.baseline import (
    ewma_forecast,
    moving_average_forecast,
    predict_ewma,
    predict_moving_average,
    predict_zero,
    zero_return_forecast,
)
from src.hk_equity.models.general_regression import (
    regression_backtest_forecast,
    regression_live_forecast,
)

from src.hk_equity.models.portfolio_router import (
    portfolio_router_backtest_forecast,
    portfolio_router_live_forecast,
)

# =============================================================================
# One adapter per model.
#
# Each adapter exposes:
# - live_forecast: returns one Series of next-week predictions;
# - backtest_forecast: returns one DataFrame of historical predictions.
#
# Model-specific fitting, feature engineering, sequence construction, and
# prediction logic belongs in the corresponding model module, not here.
# =============================================================================
MODEL_REGISTRY: dict[str, ModelAdapter] = {
    "zero": ModelAdapter(
        live_forecast=predict_zero,
        backtest_forecast=zero_return_forecast,
    ),
    "moving_average": ModelAdapter(
        live_forecast=predict_moving_average,
        backtest_forecast=moving_average_forecast,
    ),
    "ewma": ModelAdapter(
        live_forecast=predict_ewma,
        backtest_forecast=ewma_forecast,
    ),
    "general_regression": ModelAdapter(
        live_forecast=regression_live_forecast,
        backtest_forecast=regression_backtest_forecast,
    ),
}


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
    registry: dict[str, ModelAdapter],
) -> str:
    """Return sorted model names for clear error messages."""

    return ", ".join(sorted(registry.keys()))


def _get_model_adapter(
    model_config: dict[str, Any],
) -> ModelAdapter:
    """Return the adapter selected by the model YAML."""

    model_name = _get_model_name(
        model_config=model_config,
    )

    if model_name not in MODEL_REGISTRY:
        available_models = _get_available_model_names(
            registry=MODEL_REGISTRY,
        )

        raise ValueError(
            f"Unknown model name: '{model_name}'. "
            f"Available models: {available_models}"
        )

    return MODEL_REGISTRY[model_name]


def get_model_forecast(
    weekly_returns: pd.DataFrame,
    model_config: dict[str, Any],
    benchmark_returns: pd.Series | None = None,
) -> pd.Series:
    """Generate one next-week predicted return for every stock."""

    model_name = _get_model_name(
        model_config=model_config,
    )

    if model_name == "portfolio_router":
        return portfolio_router_live_forecast(
            weekly_returns=weekly_returns,
            model_config=model_config,
            benchmark_returns=benchmark_returns,
            child_forecast=get_model_forecast,
        )

    adapter = _get_model_adapter(
        model_config=model_config,
    )

    return adapter.live_forecast(
        weekly_returns=weekly_returns,
        model_config=model_config,
        benchmark_returns=benchmark_returns,
    )


def get_backtest_forecasts(
    weekly_returns: pd.DataFrame,
    model_config: dict[str, Any],
    benchmark_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Generate historical one-step-ahead forecasts for every stock and week."""

    model_name = _get_model_name(
        model_config=model_config,
    )

    if model_name == "portfolio_router":
        return portfolio_router_backtest_forecast(
            weekly_returns=weekly_returns,
            model_config=model_config,
            benchmark_returns=benchmark_returns,
            child_backtest_forecast=get_backtest_forecasts,
        )

    adapter = _get_model_adapter(
        model_config=model_config,
    )

    return adapter.backtest_forecast(
        weekly_returns=weekly_returns,
        model_config=model_config,
        benchmark_returns=benchmark_returns,
    )


def get_model_label(
    model_config: dict[str, Any],
) -> str:
    """Return the configured readable model label for outputs.

    The label is stored under ``model.label`` in each model YAML file.
    If it is missing, this function falls back to ``model.name``.
    """

    model_settings = model_config["model"]

    return model_settings.get(
        "label",
        model_settings["name"],
    )