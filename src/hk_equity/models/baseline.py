"""Define historical baseline forecasting models for weekly-return backtests.

Each function returns a full DataFrame of one-step-ahead historical forecasts.
All functions use the same interface:

    forecast_function(weekly_returns, parameters)

The output aligns forecasts with realised weekly returns. Therefore a forecast
at week t only uses information available before week t.
"""

from __future__ import annotations

import pandas as pd


def zero_return_forecast(
    weekly_returns: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Return a 0% one-step-ahead historical forecast for every stock and week.

    The zero-return baseline assumes the next weekly return is always 0%.
    ``parameters`` is accepted for a consistent model interface but is not used.

    Args:
        weekly_returns: Historical weekly-return table. Rows are week-ending
            dates and columns are ticker symbols.
        parameters: Model settings from YAML. Unused by this model.

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
    parameters: dict,
) -> pd.DataFrame:
    """Return one-step-ahead moving-average historical forecasts.

    For each stock and week, this model calculates the mean return over the
    preceding ``lookback_weeks`` completed weeks. ``shift(1)`` aligns the
    average with the following week so the forecast never uses that week's
    realised return.

    Args:
        weekly_returns: Historical weekly-return table. Rows are week-ending
            dates and columns are ticker symbols.
        parameters: Model settings from YAML. Must include
            ``lookback_weeks``.

    Returns:
        A DataFrame of historical moving-average forecasts. Initial rows are
        NaN until enough returns exist for the configured window.

    Raises:
        KeyError: If ``lookback_weeks`` is missing from ``parameters``.
    """

    lookback_weeks = parameters["lookback_weeks"]

    return (
        weekly_returns
        .rolling(window=lookback_weeks)
        .mean()
        .shift(1) #This week mean become next week mean
    )


def ewma_forecast(
    weekly_returns: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Return one-step-ahead EWMA historical forecasts.

    For each stock and week, this model calculates an exponentially weighted
    mean of historical returns. More recent returns receive higher weights.
    ``shift(1)`` aligns the result with the next week and prevents
    look-ahead bias.

    Args:
        weekly_returns: Historical weekly-return table. Rows are week-ending
            dates and columns are ticker symbols.
        parameters: Model settings from YAML. Must include ``span_weeks``.

    Returns:
        A DataFrame of historical EWMA forecasts. The first row is NaN after
        shifting because no previous-period forecast exists.

    Raises:
        KeyError: If ``span_weeks`` is missing from ``parameters``.
    """

    span_weeks = parameters["span_weeks"]

    return (
        weekly_returns                          # R(t) = a*X(t) + (1-a)*R(t-1)
        .ewm(span=span_weeks, adjust=False)     # Balance the sensitivity of the MV model
        .mean()
        .shift(1)
    )