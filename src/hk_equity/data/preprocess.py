from __future__ import annotations

import pandas as pd


def to_weekly_close(
    daily_close: pd.DataFrame,
    weekly_frequency: str = "W-FRI",
) -> pd.DataFrame:
    """Resample daily Close prices to weekly close prices.

    For ``W-FRI``, each observation represents the final available trading-day
    Close within the Monday-to-Friday period. If Friday is a trading holiday,
    pandas keeps the final available Close earlier in that same week.

    Args:
        daily_close: Daily Close prices with a DatetimeIndex and ticker columns.
        weekly_frequency: Pandas resample frequency. Default is Friday-ending
            weeks, ``W-FRI``.

    Returns:
        Weekly Close prices, excluding weeks where all ticker prices are missing.
    """

    weekly_close = (
        daily_close
        .resample(weekly_frequency) #Resample days as a group (e.g. Monday-Friday)
        .last()                     #Get the last trading day in the week
    )
    
    return weekly_close.dropna(how="all")


def calculate_returns(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate simple percentage returns.

    Formula:
        return_t = (price_t / price_(t-1)) - 1

    This function works for daily prices or weekly prices.

    Args:
        prices: Price table with a DatetimeIndex.

    Returns:
        Period-over-period percentage returns. The first row is NaN because
        no prior price exists.
    """
    #If previous = NA, then let the result be NA
#If pass weekly close, then it would become weekly return
    return prices.pct_change(fill_method=None)


def future_return_target(
    daily_close: pd.DataFrame,
    horizon_trading_days: int = 5,
) -> pd.DataFrame:
    """Calculate a future multi-trading-day return target.

    Formula:
        future_return_t = (price_(t+horizon) / price_t) - 1

    This is intended for future supervised-learning models such as Ridge,
    LightGBM, and LSTM. The final ``horizon_trading_days`` rows are NaN
    because future prices are unavailable.

    Args:
        daily_close: Daily Close-price table.
        horizon_trading_days: Forecast horizon in trading days. Five is the
            default approximation of one trading week.

    Returns:
        Forward return target table.
    """

    future_return = (
        daily_close.shift(-horizon_trading_days)    #5 days later
        / daily_close                               #today
        - 1                                         #get the future return
    )

    return future_return