from __future__ import annotations

import numpy as np
import pandas as pd


def forecast_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict:
    """Calculate forecast-error and directional-accuracy metrics.

    Args:
        actual: Actual realised returns.
        predicted: Model predicted returns.

    Returns:
        Dictionary containing:
        - observations
        - MAE
        - RMSE
        - directional_accuracy
        - mean_predicted_return
        - mean_actual_return
        - forecast_bias
    """

    valid = pd.concat(
        [
            actual.rename("actual"),
            predicted.rename("predicted"),
        ],
        axis=1,
    ).dropna()

    if valid.empty:
        return {
            "observations": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "directional_accuracy": np.nan,
            "mean_predicted_return": np.nan,
            "mean_actual_return": np.nan,
            "forecast_bias": np.nan,
        }

    error = valid["predicted"] - valid["actual"]

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

    For every week:
        Rank(predicted return across stocks)
        versus
        Rank(actual return across stocks)

    A positive IC means stocks ranked highly by the model tended to have
    relatively stronger realised returns during that week.
    """

    common_dates = actual_returns.index.intersection(
        predicted_returns.index
    )

    records = []

    for date in common_dates:
        comparison = pd.concat(
            [
                actual_returns.loc[date].rename("actual"),
                predicted_returns.loc[date].rename("predicted"),
            ],
            axis=1,
        ).dropna()

        if len(comparison) < 2:
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
    """Calculate realised average return of the model's Top-N predicted stocks."""

    common_dates = actual_returns.index.intersection(
        predicted_returns.index
    )

    records = []

    for date in common_dates:
        comparison = pd.concat(
            [
                actual_returns.loc[date].rename("actual"),
                predicted_returns.loc[date].rename("predicted"),
            ],
            axis=1,
        ).dropna()

        if len(comparison) < top_n:
            continue

        top_stocks = comparison.nlargest(
            top_n,
            "predicted",
        )

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