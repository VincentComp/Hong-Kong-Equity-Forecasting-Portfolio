import pandas as pd


def zero_return_forecast(
    weekly_returns: pd.DataFrame
) -> pd.DataFrame:
    """Generates a zero-return forecast baseline.

    Implements a naive random walk benchmark that assumes the expected return
    for all assets in the next period is exactly 0%.

    Args:
        weekly_returns (pd.DataFrame): Historical weekly returns with a 
            DatetimeIndex and ticker symbols as column headers.

    Returns:
        pd.DataFrame: A DataFrame of identical shape filled with zeros.
    """
    return weekly_returns.copy() * 0


def moving_average_forecast(
    weekly_returns: pd.DataFrame,
    lookback_weeks: int = 4,
) -> pd.DataFrame:
    """Computes a Simple Moving Average (SMA) return forecast.

    Forecasts the next week's return as the unweighted arithmetic mean over the
    preceding `lookback_weeks`. The output is shifted forward by 1 period to
    prevent look-ahead bias (data leakage).

    Args:
        weekly_returns (pd.DataFrame): Historical weekly returns with a 
            DatetimeIndex and ticker symbols as column headers.
        lookback_weeks (int, optional): The rolling window length in weeks. 
            Defaults to 4.

    Returns:
        pd.DataFrame: One-step-ahead simple moving average forecasts.
    """
    forecast = (
        weekly_returns
        .rolling(lookback_weeks)
        .mean()
        .shift(1) #This week mean become next week mean
    )

    return forecast


def ewma_forecast(
    weekly_returns: pd.DataFrame,
    span_weeks: int = 4,
) -> pd.DataFrame:
    """Computes an Exponentially Weighted Moving Average (EWMA) return forecast.

    Forecasts the next week's return using exponentially decaying weights, giving
    higher importance to recent market observations. The output is shifted 
    forward by 1 period to avoid look-ahead bias.

    The decay factor alpha is calculated as:
        alpha = 2 / (span_weeks + 1)

    Args:
        weekly_returns (pd.DataFrame): Historical weekly returns with a 
            DatetimeIndex and ticker symbols as column headers.
        span_weeks (int, optional): The exponential decay window span in weeks. 
            Defaults to 4.

    Returns:
        pd.DataFrame: One-step-ahead exponentially weighted moving average forecasts.
    """
    forecast = (                                # a*R(T) + (1-a)*R(t-1)
        weekly_returns                          # Balance the sensitivity of the MV model
        .ewm(span=span_weeks, adjust=False) 
        .mean()
        .shift(1)
    )

    return forecast
    
