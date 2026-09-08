import pandas as pd


def to_weekly_close(
    daily_close: pd.DataFrame
) -> pd.DataFrame:
    weekly_close = (
        daily_close
        .resample("W-FRI")
        .last()
    )

    return weekly_close.dropna(how="all")


def calculate_returns(
    prices: pd.DataFrame
) -> pd.DataFrame:
    return prices.pct_change(fill_method=None)


def future_weekly_return_target(
    daily_close: pd.DataFrame,
    horizon_days: int = 5,
) -> pd.DataFrame:
    future_return = (
        daily_close.shift(-horizon_days)
        / daily_close
        - 1
    )

    return future_return