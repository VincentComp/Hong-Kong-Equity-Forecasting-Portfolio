"""Calculate performance metrics for historical weekly-return forecasts.

This module evaluates how closely a model's predicted weekly returns match
the realised weekly returns. It provides:

- Point forecast metrics: MAE, RMSE, forecast bias, and directional accuracy.
- Cross-sectional ranking metrics: weekly Spearman Rank IC.
- Simple stock-selection evaluation: realised return of the model's Top-N
  predicted stocks versus an equal-weight benchmark.

These functions are mainly called by ``scripts/run_backtest.py`` after a model
has generated historical one-step-ahead forecasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forecast_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict:
    """Calculate point-forecast error, direction, and bias metrics.

    The function first aligns the actual and predicted return Series by index
    and removes rows where either value is missing. This is important because
    some early backtest weeks can have no forecast, for example before a
    moving-average model has enough historical observations.

    Metrics returned:
    - ``observations``: Number of valid actual-predicted return pairs.
    - ``MAE``: Mean Absolute Error. Average size of forecast errors, ignoring
      whether the forecast was too high or too low. Lower is better.
    - ``RMSE``: Root Mean Squared Error. Like MAE, but assigns a larger penalty
      to large forecast errors. Lower is better.
    - ``directional_observations``: Number of valid pairs with non-zero actual
      and predicted returns used for directional accuracy.
    - ``directional_accuracy``: Proportion of non-zero predictions that have
      the same sign as the realised return. For example, positive prediction
      and positive actual return counts as correct.
    - ``mean_predicted_return``: Average predicted return over valid rows.
    - ``mean_actual_return``: Average realised return over valid rows.
    - ``forecast_bias``: Average signed error, calculated as
      predicted return minus actual return. A positive value means the model
      tends to over-predict; a negative value means it tends to under-predict.

    A zero-return baseline has no non-zero forecasts. Therefore its
    ``directional_accuracy`` will be ``NaN`` rather than treating a 0% forecast
    as either an up or down prediction.

    Args:
        actual: Series of realised weekly returns. Its index should match the
            corresponding dates or observations in ``predicted``.
        predicted: Series of model-predicted weekly returns.

    Returns:
        A dictionary containing forecast-error, direction, average-return, and
        forecast-bias metrics. If no valid actual-predicted pairs exist, metric
        values are returned as ``numpy.nan`` and ``observations`` is zero.

    Example:
        >>> actual = pd.Series([0.01, -0.02, 0.03])
        >>> predicted = pd.Series([0.02, -0.01, 0.01])
        >>> results = forecast_metrics(actual, predicted)
        >>> results["observations"]
        3
    """

    valid = pd.concat(
        [
            actual.rename("actual"),
            predicted.rename("predicted"),
        ],
        axis=1,
    ).dropna() #drop the row if na exist

    if valid.empty: #Return all na for empty df
        return {
            "observations": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "directional_accuracy": np.nan,
            "mean_predicted_return": np.nan,
            "mean_actual_return": np.nan,
            "forecast_bias": np.nan,
        }

    #Calculate the error
    error = valid["predicted"] - valid["actual"]

    #Calculate the diretional accuracy
    directional_valid = valid[
        (valid["predicted"] != 0)
        & (valid["actual"] != 0)
    ].copy()

    if directional_valid.empty:
        directional_accuracy = np.nan
    else:
        directional_accuracy = (
            np.sign(directional_valid["predicted"])
            == np.sign(directional_valid["actual"])
        ).mean()

    return {
        "observations": len(valid),
        "MAE": error.abs().mean(),
        "RMSE": np.sqrt((error ** 2).mean()),
        "directional_observations": len(directional_valid),
        "directional_accuracy": directional_accuracy,
        "mean_predicted_return": valid["predicted"].mean(),
        "mean_actual_return": valid["actual"].mean(),
        "forecast_bias": error.mean(),
    }


def calculate_rank_ic_by_week(
    actual_returns: pd.DataFrame,
    predicted_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate weekly cross-sectional Spearman Rank Information Coefficient.

    For each available week, this function compares the model's ranking of
    stocks with the realised ranking of stocks:

    ```text
    Predicted returns across all stocks
    versus
    Actual realised returns across all stocks
    ```

    It uses Spearman correlation, which compares ranks rather than the exact
    numerical values. This is useful when the project cares about identifying
    which stocks may perform relatively better or worse, even if the predicted
    percentage returns are not perfectly accurate.

    Rank IC interpretation:
    - Positive Rank IC: Stocks given higher predicted returns generally had
      higher realised returns that week.
    - Negative Rank IC: The model tended to rank stocks in the wrong order.
    - Rank IC near zero: The model showed little useful cross-sectional ranking
      relationship for that week.
    - ``NaN``: Cannot calculate a meaningful ranking. For example, a
      zero-return baseline gives every stock the same prediction.

    The function excludes stocks with missing actual or predicted values for
    each week. At least two valid stocks are needed to calculate a correlation.

    Mean Rank IC > 0.03 (3%)    :   Ok
    Mean Rank IC > 0.05 (5%)    :   Good
    Mean Rank IC > 0.10 (10%)   :   Data Leakage (?)

    Args:
        actual_returns: Wide DataFrame of realised weekly returns. Rows are
            week-ending dates and columns are ticker symbols.
        predicted_returns: Wide DataFrame of predicted weekly returns, aligned
            with ``actual_returns`` by dates and ticker symbols.

    Returns:
        A DataFrame with one row per eligible week containing:
        - ``week_ending``: Week-ending date.
        - ``number_of_stocks``: Number of valid stocks used in that week's
          rank comparison.
        - ``rank_ic``: Spearman rank correlation between predicted and actual
          stock returns.

    Example:
        >>> rank_ic_table = calculate_rank_ic_by_week(
        ...     actual_returns=actual_returns,
        ...     predicted_returns=predicted_returns,
        ... )
        >>> rank_ic_table[["week_ending", "rank_ic"]].head()
    """

    common_dates = actual_returns.index.intersection(
        predicted_returns.index
    )#get the date that both actual and predict exist

    records = []

    for date in common_dates:
        comparison = pd.concat(
            [
                actual_returns.loc[date].rename("actual"),
                predicted_returns.loc[date].rename("predicted"),
            ],
            axis=1,
        ).dropna()#drop the row with stock having na

        if len(comparison) < 2: #require at least 2 stocks
            continue

        # A constant forecast, such as zero-return baseline, has no ranking.
        if comparison["predicted"].nunique() < 2:
            rank_ic = np.nan
        else:
            rank_ic = comparison["predicted"].corr(
                comparison["actual"],
                method="spearman",
            )

        records.append({
            "week_ending": date,
            "number_of_stocks": len(comparison),
            "rank_ic": rank_ic,
        })

    return pd.DataFrame(records)


def calculate_top_n_return(
    actual_returns: pd.DataFrame,
    predicted_returns: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """Evaluate the realised return of the model-selected Top-N stocks.

    For each week, this function selects the ``top_n`` stocks with the highest
    model-predicted returns. It then calculates the average realised return of
    those selected stocks and compares it with the average realised return of
    all valid stocks in the same week.

    This is a simple equal-weighted stock-selection evaluation. It answers:

    ```text
    If we bought the stocks that the model ranked highest,
    did they outperform the equal-weight portfolio of all available stocks?
    ```

    The function does not account for transaction costs, trading constraints,
    volatility, position limits, or rebalancing costs. It is therefore a
    ranking diagnostic rather than a complete portfolio backtest.

    A constant prediction cannot rank stocks. Therefore, weeks where every
    predicted return is identical, such as the zero-return baseline, are
    excluded from the output.

    Args:
        actual_returns: Wide DataFrame of realised weekly returns. Rows are
            week-ending dates and columns are ticker symbols.
        predicted_returns: Wide DataFrame of predicted weekly returns, aligned
            with ``actual_returns`` by dates and ticker symbols.
        top_n: Number of highest-ranked stocks to select each week. Default is
            3.

    Returns:
        A DataFrame with one row per eligible week containing:
        - ``week_ending``: Week-ending date.
        - ``top_n``: Number of selected highest-predicted-return stocks.
        - ``top_n_average_actual_return``: Equal-weight average realised return
          of the selected Top-N stocks.
        - ``equal_weight_average_actual_return``: Equal-weight average realised
          return across every valid stock that week.
        - ``top_n_minus_equal_weight``: Difference between selected Top-N
          return and the equal-weight benchmark. Positive means the model's
          Top-N selection outperformed in that week.

    Example:
        >>> top_3_table = calculate_top_n_return(
        ...     actual_returns=actual_returns,
        ...     predicted_returns=predicted_returns,
        ...     top_n=3,
        ... )
        >>> top_3_table["top_n_minus_equal_weight"].mean()
    """

    common_dates = actual_returns.index.intersection(
        predicted_returns.index
    ) #find the dates that predict and actual exist

    records = []

    for date in common_dates:
        comparison = pd.concat(
            [
                actual_returns.loc[date].rename("actual"),
                predicted_returns.loc[date].rename("predicted"),
            ],
            axis=1,
        ).dropna() #Drop the row with stock having na

        if len(comparison) < top_n:
            continue

        top_stocks = comparison.nlargest(
            top_n,
            "predicted",
        )

        #Skip the constant model
        if comparison["predicted"].nunique() < 2:
            continue

        records.append({
            "week_ending": date,
            "top_n": top_n,
            "top_n_average_actual_return": (
                top_stocks["actual"].mean()
            ),
            "equal_weight_average_actual_return": (
                comparison["actual"].mean()
            ),
            "top_n_minus_equal_weight": (
                top_stocks["actual"].mean()
                - comparison["actual"].mean()
            ),
        })

    return pd.DataFrame(records)