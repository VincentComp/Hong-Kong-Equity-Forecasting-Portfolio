import pandas as pd


def zero_return_forecast(
    weekly_returns: pd.DataFrame
) -> pd.DataFrame:
    return weekly_returns.copy() * 0


def moving_average_forecast(
    weekly_returns: pd.DataFrame,
    lookback_weeks: int = 4,
) -> pd.DataFrame:
    forecast = (
        weekly_returns
        .rolling(lookback_weeks)
        .mean()
        .shift(1)
    )

    return forecast


def ewma_forecast(
    weekly_returns: pd.DataFrame,
    span_weeks: int = 4,
) -> pd.DataFrame:
    forecast = (
        weekly_returns
        .ewm(span=span_weeks, adjust=False)
        .mean()
        .shift(1)
    )

    return forecast