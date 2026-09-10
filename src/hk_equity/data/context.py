from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.hk_equity.data.preprocess import (
    calculate_returns,
    to_weekly_close,
)


@dataclass
class ForecastContext:
    """Common time and data information required by every forecast model."""

    daily_close: pd.DataFrame
    weekly_close: pd.DataFrame
    weekly_returns: pd.DataFrame

    data_cutoff: pd.Timestamp           #--as-of date (theorectical last date set by the user)
    latest_data_date: pd.Timestamp      #actual last day in the dataset
    latest_completed_week: pd.Timestamp #actual last day (in a complete week) in the dataset

    target_week_start: pd.Timestamp     #forecasting period
    target_week_end: pd.Timestamp


def build_forecast_context(
    daily_close: pd.DataFrame,
    data_cutoff: pd.Timestamp,
    weekly_frequency: str = "W-FRI",
    horizon_trading_days: int = 5,
) -> ForecastContext:
    """Create one common forecast context from a historical data snapshot.

    The function ensures that all models use exactly the same:

    - data cutoff;
    - weekly price definition;
    - completed weekly returns;
    - next forecast-week date range.

    Args:
        daily_close: Full daily Close-price history.
        data_cutoff: Latest date whose information the model may use.
        weekly_frequency: Weekly aggregation rule, normally ``W-FRI``.
        horizon_trading_days: Expected number of trading days in the target week.

    Returns:
        ForecastContext containing cleaned data and all relevant dates.
    """

    # Keep only data that was available on or before the forecast cutoff date.
    usable_daily_close = (#get the dates before deadline
        daily_close
        .loc[:data_cutoff]
        .dropna(how="all")
    )

    if usable_daily_close.empty:
        raise ValueError(
            "No daily Close prices exist on or before the requested data cutoff."
        )


    # This may differ from data_cutoff if the cutoff is a weekend or public holiday.
    latest_data_date = usable_daily_close.index.max()

    weekly_close = to_weekly_close(
        daily_close=usable_daily_close,
        weekly_frequency=weekly_frequency,
    )

    # Exclude a partially completed week from forecast inputs.
    weekly_close = weekly_close.loc[
        weekly_close.index <= data_cutoff.normalize()
    ]

    weekly_returns = (
        calculate_returns(weekly_close)
        .dropna(how="all")
    )

    if weekly_returns.empty:
        raise ValueError(
            "No completed weekly returns could be calculated from the input data."
        )

    latest_completed_week = weekly_returns.index.max()



    # The Project 1 target is the next Friday-ending week.
    target_week_end = (
        latest_completed_week
        + pd.Timedelta(days=7)
    )

    target_week_start = (
        target_week_end
        - pd.Timedelta(days=horizon_trading_days - 1)
    )

    return ForecastContext(
        daily_close=usable_daily_close,
        weekly_close=weekly_close,
        weekly_returns=weekly_returns,
        data_cutoff=data_cutoff,
        latest_data_date=latest_data_date,
        latest_completed_week=latest_completed_week,
        target_week_start=target_week_start,
        target_week_end=target_week_end,
    )