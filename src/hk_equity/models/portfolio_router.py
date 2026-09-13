"""Validate and prepare portfolio-level model routing settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.hk_equity.utils.config import (
    load_yaml,
)


def load_portfolio_router_plan(
    model_config: dict[str, Any],
    portfolio_tickers: list[str],
) -> dict[str, Any]:
    """Load and validate a portfolio router model configuration.

    The router assigns exactly one child model profile to every portfolio
    ticker. The benchmark ticker is not part of portfolio_tickers and must
    therefore never appear in assignments.
    """

    profiles = model_config.get(
        "profiles",
        {},
    )

    assignments = model_config.get(
        "assignments",
        {},
    )

    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(
            "portfolio_router config must contain "
            "a non-empty profiles mapping."
        )

    if not isinstance(assignments, dict) or not assignments:
        raise ValueError(
            "portfolio_router config must contain "
            "a non-empty assignments mapping."
        )

    expected_tickers = set(
        str(ticker)
        for ticker in portfolio_tickers
    )

    assigned_tickers = set(
        str(ticker)
        for ticker in assignments
    )

    missing_tickers = sorted(
        expected_tickers - assigned_tickers
    )

    unexpected_tickers = sorted(
        assigned_tickers - expected_tickers
    )

    if missing_tickers or unexpected_tickers:
        raise ValueError(
            "portfolio_router assignments must contain "
            "each portfolio ticker exactly once. "
            f"Missing: {missing_tickers}; "
            f"unexpected: {unexpected_tickers}"
        )

    resolved_profiles: dict[str, dict[str, Any]] = {}

    for profile_name, profile_settings in profiles.items():
        profile_name = str(profile_name)

        if not isinstance(profile_settings, dict):
            raise TypeError(
                f"Profile '{profile_name}' must be a mapping."
            )

        config_path_value = profile_settings.get(
            "config_path",
        )

        if not config_path_value:
            raise ValueError(
                f"Profile '{profile_name}' must contain "
                "config_path."
            )

        config_path = Path(
            str(config_path_value)
        )

        if not config_path.exists():
            raise FileNotFoundError(
                f"Profile '{profile_name}' config file "
                f"does not exist: {config_path}"
            )

        child_model_config = load_yaml(
            config_path,
        )

        child_model_settings = child_model_config.get(
            "model",
            {},
        )

        child_model_name = str(
            child_model_settings.get(
                "name",
                "",
            )
        ).lower()

        if not child_model_name:
            raise ValueError(
                f"Profile '{profile_name}' child config "
                "must contain model.name."
            )

        if child_model_name == "portfolio_router":
            raise ValueError(
                f"Profile '{profile_name}' cannot reference "
                "another portfolio_router config."
            )

        resolved_profiles[profile_name] = {
            "config_path": str(config_path),
            "model_config": child_model_config,
        }

    resolved_assignments: dict[str, str] = {}

    for ticker, profile_name in assignments.items():
        ticker = str(ticker)
        profile_name = str(profile_name)

        if profile_name not in resolved_profiles:
            raise ValueError(
                f"Ticker '{ticker}' references unknown "
                f"profile '{profile_name}'."
            )

        resolved_assignments[ticker] = profile_name

    return {
        "profiles": resolved_profiles,
        "assignments": resolved_assignments,
    }

def portfolio_router_live_forecast(
    weekly_returns: pd.DataFrame,
    model_config: dict[str, Any],
    benchmark_returns: pd.Series | None,
    child_forecast: Callable[
        [
            pd.DataFrame,
            dict[str, Any],
            pd.Series | None,
        ],
        pd.Series,
    ],
) -> pd.Series:
    """Combine live forecasts from different child model profiles.

    Each unique child model is run once. The router then selects only the
    configured ticker predictions from that child model and combines them
    into one complete portfolio forecast.
    """

    portfolio_tickers = list(
        weekly_returns.columns
    )

    router_plan = load_portfolio_router_plan(
        model_config=model_config,
        portfolio_tickers=portfolio_tickers,
    )

    profile_predictions: dict[str, pd.Series] = {}

    for profile_name, profile_details in (
        router_plan["profiles"].items()
    ):
        child_prediction = child_forecast(
            weekly_returns=weekly_returns,
            model_config=profile_details[
                "model_config"
            ],
            benchmark_returns=benchmark_returns,
        )

        child_prediction = child_prediction.reindex(
            portfolio_tickers
        )

        profile_predictions[profile_name] = (
            child_prediction
        )

    routed_prediction = pd.Series(
        index=portfolio_tickers,
        dtype=float,
        name="predicted_weekly_return",
    )

    for ticker in portfolio_tickers:
        profile_name = router_plan["assignments"][
            ticker
        ]

        predicted_value = profile_predictions[
            profile_name
        ].loc[ticker]

        if pd.isna(predicted_value):
            raise ValueError(
                f"Profile '{profile_name}' produced no "
                f"live prediction for '{ticker}'."
            )

        routed_prediction.loc[ticker] = float(
            predicted_value
        )

    return routed_prediction


def portfolio_router_backtest_forecast(
    weekly_returns: pd.DataFrame,
    model_config: dict[str, Any],
    benchmark_returns: pd.Series | None,
    child_backtest_forecast: Callable[
        [
            pd.DataFrame,
            dict[str, Any],
            pd.Series | None,
        ],
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    """Combine backtest forecasts from different child model profiles.

    Every child model produces its historical forecast table once. The router
    takes one assigned ticker column from each profile and returns a complete
    portfolio prediction matrix.
    """

    portfolio_tickers = list(
        weekly_returns.columns
    )

    router_plan = load_portfolio_router_plan(
        model_config=model_config,
        portfolio_tickers=portfolio_tickers,
    )

    profile_predictions: dict[str, pd.DataFrame] = {}

    for profile_name, profile_details in (
        router_plan["profiles"].items()
    ):
        child_predictions = child_backtest_forecast(
            weekly_returns=weekly_returns,
            model_config=profile_details[
                "model_config"
            ],
            benchmark_returns=benchmark_returns,
        )

        missing_columns = [
            ticker
            for ticker in portfolio_tickers
            if ticker not in child_predictions.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Profile '{profile_name}' backtest is "
                "missing portfolio prediction columns: "
                f"{missing_columns}"
            )

        profile_predictions[profile_name] = (
            child_predictions.reindex(
                index=weekly_returns.index,
                columns=portfolio_tickers,
            )
        )

    routed_predictions = pd.DataFrame(
        index=weekly_returns.index,
        columns=portfolio_tickers,
        dtype=float,
    )

    for ticker in portfolio_tickers:
        profile_name = router_plan["assignments"][
            ticker
        ]

        routed_predictions[ticker] = (
            profile_predictions[profile_name][ticker]
        )

    return routed_predictions