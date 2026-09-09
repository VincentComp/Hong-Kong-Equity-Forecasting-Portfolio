import pandas as pd


def to_weekly_close(
    daily_close: pd.DataFrame
) -> pd.DataFrame:
    """Resamples daily close prices to a weekly Friday frequency.

    Aggregates daily equity price data by extracting the last available trading
    day price within each calendar week ending on Friday (`W-FRI`). If Friday is
    a trading holiday, the last preceding valid trading price of that week is used.

    Args:
        daily_close (pd.DataFrame): Daily close prices with a DatetimeIndex 
            and ticker symbols as column headers.

    Returns:
        pd.DataFrame: Weekly close prices resampled to Friday frequencies, 
            with fully empty holiday/closure weeks dropped.
    """
    weekly_close = (
        daily_close
        .resample("W-FRI")  #Resample Monday-Friday as a group
        .last()             #Get the last trading day in the week
    )

    return weekly_close.dropna(how="all") #remove the entire-empty row (e.g. New Year Holiday)


#If previous = NA, then let the result be NA
#If pass weekly close, then it would become weekly return
def calculate_returns(
    prices: pd.DataFrame
) -> pd.DataFrame:
    """Calculates percentage price returns between consecutive time periods.

    Computes the simple period-over-period percentage change:
        Return_t = (Price_t - Price_{t-1}) / Price_{t-1}

    Args:
        prices (pd.DataFrame): Historical price time series (daily or weekly) 
            with a DatetimeIndex.

    Returns:
        pd.DataFrame: Percentage returns for each ticker. The initial row 
            will contain NaN values due to lag differencing.
    """
    return prices.pct_change(fill_method=None) 


def future_weekly_return_target(
    daily_close: pd.DataFrame,
    horizon_days: int = 5,
) -> pd.DataFrame:
    """Calculates forward multi-day return targets for supervised learning.

    Computes the future percentage price change over a specified holding horizon:
        Future_Return_t = (Price_{t + horizon} / Price_t) - 1

    This shifts future prices backward to pair current-day feature vectors (X_t)
    with forward-looking targets (y_t) without causing historical data leakage.

    Args:
        daily_close (pd.DataFrame): Daily close prices with a DatetimeIndex.
        horizon_days (int, optional): The forward holding period in trading days. 
            Defaults to 5 (representing 1 trading week).

    Returns:
        pd.DataFrame: Forward return targets. The last `horizon_days` rows 
            will contain NaN values as future data is not yet available.
    """
    future_return = (
        daily_close.shift(-horizon_days)    #5 days later
        / daily_close                       #today
        - 1                                 #Get the future return 
    )

    return future_return