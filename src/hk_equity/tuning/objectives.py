"""Objective functions for Optuna model-tuning studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.hk_equity.evaluation.metrics import (
    forecast_metrics,
)


ScoreFunction = Callable[
    [pd.Series, pd.Series, dict],
    float,
]


@dataclass(frozen=True)
class ObjectiveSpec:
    """Describe one supported Optuna objective."""

    name: str
    direction: str
    score_function: ScoreFunction


def _align_actual_and_predicted(
    actual: pd.Series,
    predicted: pd.Series,
) -> pd.DataFrame:
    """Align actual and predicted values and remove missing pairs."""

    aligned = pd.concat(
        [
            actual.rename("actual"),
            predicted.rename("predicted"),
        ],
        axis=1,
    ).dropna()

    if aligned.empty:
        raise ValueError(
            "No valid actual/predicted pairs are available "
            "for the tuning objective."
        )

    return aligned


def score_mae(
    actual: pd.Series,
    predicted: pd.Series,
    settings: dict,
) -> float:
    """Return base-stock Mean Absolute Error."""

    del settings

    metrics = forecast_metrics(
        actual=actual,
        predicted=predicted,
    )

    score = metrics["MAE"]

    if pd.isna(score):
        raise ValueError(
            "MAE cannot be calculated for the tuning objective."
        )

    return float(score)


def score_directional_accuracy(
    actual: pd.Series,
    predicted: pd.Series,
    settings: dict,
) -> float:
    """Return base-stock directional accuracy."""

    del settings

    metrics = forecast_metrics(
        actual=actual,
        predicted=predicted,
    )

    score = metrics["directional_accuracy"]

    if pd.isna(score):
        raise ValueError(
            "Directional accuracy cannot be calculated "
            "for the tuning objective."
        )

    return float(score)


def score_time_series_spearman(
    actual: pd.Series,
    predicted: pd.Series,
    settings: dict,
) -> float:
    """Return time-series Spearman correlation for one base stock."""

    del settings

    aligned = _align_actual_and_predicted(
        actual=actual,
        predicted=predicted,
    )

    score = aligned["predicted"].corr(
        aligned["actual"],
        method="spearman",
    )

    if pd.isna(score):
        raise ValueError(
            "Time-series Spearman correlation cannot be calculated."
        )

    return float(score)


def score_long_only_sharpe(
    actual: pd.Series,
    predicted: pd.Series,
    settings: dict,
) -> float:
    """Return annualized Sharpe ratio for a prediction-based long-only rule.

    The strategy is:
    - Go long the base stock when predicted return is positive.
    - Hold cash when predicted return is zero or negative.
    - Cash return is assumed to be zero.
    """

    annualization_factor = int(
        settings.get(
            "annualization_factor",
            52,
        )
    )

    transaction_cost_bps = float(
        settings.get(
            "transaction_cost_bps",
            0.0,
        )
    )

    if annualization_factor <= 0:
        raise ValueError(
            "annualization_factor must be greater than zero."
        )

    if transaction_cost_bps < 0:
        raise ValueError(
            "transaction_cost_bps cannot be negative."
        )

    aligned = _align_actual_and_predicted(
        actual=actual,
        predicted=predicted,
    )

    position = (
        aligned["predicted"] > 0
    ).astype(float)

    turnover = position.diff().abs().fillna(
        position.abs()
    )

    transaction_cost_rate = (
        transaction_cost_bps / 10_000
    )

    strategy_returns = (
        position * aligned["actual"]
        - turnover * transaction_cost_rate
    )

    strategy_volatility = strategy_returns.std(
        ddof=1
    )

    if (
        pd.isna(strategy_volatility)
        or strategy_volatility == 0
    ):
        raise ValueError(
            "Sharpe ratio cannot be calculated because "
            "strategy-return volatility is zero."
        )

    return float(
        np.sqrt(annualization_factor)
        * strategy_returns.mean()
        / strategy_volatility
    )


OBJECTIVE_REGISTRY: dict[str, ObjectiveSpec] = {
    "mae": ObjectiveSpec(
        name="mae",
        direction="minimize",
        score_function=score_mae,
    ),
    "directional_accuracy": ObjectiveSpec(
        name="directional_accuracy",
        direction="maximize",
        score_function=score_directional_accuracy,
    ),
    "time_series_spearman": ObjectiveSpec(
        name="time_series_spearman",
        direction="maximize",
        score_function=score_time_series_spearman,
    ),
    "long_only_sharpe": ObjectiveSpec(
        name="long_only_sharpe",
        direction="maximize",
        score_function=score_long_only_sharpe,
    ),
}


def get_objective_spec(
    objective_name: str,
) -> ObjectiveSpec:
    """Return one validated objective specification."""

    normalized_name = str(
        objective_name
    ).lower()

    if normalized_name not in OBJECTIVE_REGISTRY:
        available_objectives = ", ".join(
            sorted(OBJECTIVE_REGISTRY)
        )

        raise ValueError(
            f"Unknown tuning objective '{objective_name}'. "
            f"Available objectives: {available_objectives}"
        )

    return OBJECTIVE_REGISTRY[
        normalized_name
    ]