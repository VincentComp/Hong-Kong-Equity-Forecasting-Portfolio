"""Define historical baseline forecasting models for weekly-return backtests.

Each function returns a full DataFrame of one-step-ahead historical forecasts.
All functions use the same interface:

    forecast_function(weekly_returns, model_config, benchmark_returns)

The adapter-compatible interface accepts the complete model configuration.
Baseline models only read ``model_config["parameters"]`` and ignore
``benchmark_returns``. The output aligns forecasts with realised weekly
returns. Therefore a forecast at week t only uses information available before
week t.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------


def _get_parameters(
    model_config: dict[str, Any],
) -> dict[str, Any]:
    """Return baseline parameters from a complete model configuration."""

    return model_config.get(
        "parameters",
        {},
    )


def _validate_positive_integer(
    value: Any,
    parameter_name: str,
) -> int:
    """Validate and return a positive integer model parameter."""

    try:
        integer_value = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{parameter_name} must be an integer."
        ) from error

    if integer_value <= 0:
        raise ValueError(
            f"{parameter_name} must be greater than zero."
        )

    return integer_value


# -----------------------------------------------------------------------------
# Live forecast functions
# -----------------------------------------------------------------------------


def predict_zero(
    weekly_returns: pd.DataFrame,
    model_config: dict[str, Any],
    benchmark_returns: pd.Series | None = None,
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
    model_config: dict[str, Any],
    benchmark_returns: pd.Series | None = None,
) -> pd.Series:
    """Predict next-week returns from the recent simple average return."""

    parameters = _get_parameters(model_config)
    lookback_weeks = _validate_positive_integer(
        parameters.get("lookback_weeks"),
        "lookback_weeks",
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
    model_config: dict[str, Any],
    benchmark_returns: pd.Series | None = None,
) -> pd.Series:
    """Predict next-week returns from an exponentially weighted average."""

    parameters = _get_parameters(model_config)
    span_weeks = _validate_positive_integer(
        parameters.get("span_weeks"),
        "span_weeks",
    )

    if len(weekly_returns) < span_weeks:
        raise ValueError(
            f"EWMA needs at least {span_weeks} "
            "completed weekly returns."
        )

    return pd.Series(
        weekly_returns
        .ewm(
            span=span_weeks,
            adjust=False,
        )
        .mean()
        .iloc[-1],
        name="predicted_weekly_return",
    )
    #a = 2 / (1+span)
    #EWMA(t) = a*R(t) + (1-a)*EWMA(t-1)


# -----------------------------------------------------------------------------
# Historical backtest functions
# -----------------------------------------------------------------------------


def zero_return_forecast(
    weekly_returns: pd.DataFrame,
    model_config: dict[str, Any],
    benchmark_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Return a 0% one-step-ahead historical forecast for every stock and week.

    The zero-return baseline assumes the next weekly return is always 0%.
    ``model_config`` is accepted for a consistent model interface but is not
    used by this model.

    Args:
        weekly_returns: Historical weekly-return table. Rows are week-ending
            dates and columns are ticker symbols.
        model_config: Complete model settings from YAML. Unused by this model.
        benchmark_returns: Optional benchmark returns. Unused by this model.

    Returns:
        A DataFrame with the same index and columns as ``weekly_returns``,
        filled with 0.0.
    """

    return pd.DataFrame(
        data=0.0,
        index=weekly_returns.index,
        columns=weekly_returns.columns,
        dtype=float,
    )


def moving_average_forecast(
    weekly_returns: pd.DataFrame,
    model_config: dict[str, Any],
    benchmark_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Return one-step-ahead moving-average historical forecasts.

    For each stock and week, this model calculates the mean return over the
    preceding ``lookback_weeks`` completed weeks. ``shift(1)`` aligns the
    average with the following week so the forecast never uses that week's
    realised return.

    Args:
        weekly_returns: Historical weekly-return table. Rows are week-ending
            dates and columns are ticker symbols.
        model_config: Complete model settings from YAML. Must contain
            ``parameters.lookback_weeks``.
        benchmark_returns: Optional benchmark returns. Unused by this model.

    Returns:
        A DataFrame of historical moving-average forecasts. Initial rows are
        NaN until enough returns exist for the configured window.
    """

    parameters = _get_parameters(model_config)
    lookback_weeks = _validate_positive_integer(
        parameters.get("lookback_weeks"),
        "lookback_weeks",
    )

    return (
        weekly_returns
        .rolling(window=lookback_weeks)
        .mean()
        .shift(1)
    )


def ewma_forecast(
    weekly_returns: pd.DataFrame,
    model_config: dict[str, Any],
    benchmark_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Return one-step-ahead EWMA historical forecasts.

    For each stock and week, this model calculates an exponentially weighted
    mean of historical returns. More recent returns receive higher weights.
    ``shift(1)`` aligns the result with the next week and prevents
    look-ahead bias.

    Args:
        weekly_returns: Historical weekly-return table. Rows are week-ending
            dates and columns are ticker symbols.
        model_config: Complete model settings from YAML. Must contain
            ``parameters.span_weeks``.
        benchmark_returns: Optional benchmark returns. Unused by this model.

    Returns:
        A DataFrame of historical EWMA forecasts. The first row is NaN after
        shifting because no previous-period forecast exists.
    """

    parameters = _get_parameters(model_config)
    span_weeks = _validate_positive_integer(
        parameters.get("span_weeks"),
        "span_weeks",
    )

    return (
        weekly_returns
        .ewm(
            span=span_weeks,
            adjust=False,
        )
        .mean()
        .shift(1)
    )