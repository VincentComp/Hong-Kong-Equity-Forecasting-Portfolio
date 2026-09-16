"""Weighted ensemble forecasts built from normal child model YAML configs.

The ensemble model combines predictions from multiple existing models.
Each component must point to a normal, non-ensemble model YAML config.

The module supports:
- Live next-week forecasts as a weighted Series.
- Historical expanding-window backtests as a weighted DataFrame.
- Full portfolio ticker validation.
- Weight validation.
- Protection against nested or self-referencing ensembles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.hk_equity.utils.config import (
    load_yaml,
)


LiveForecastFunction = Callable[
    [
        pd.DataFrame,
        dict[str, Any],
        pd.Series | None,
    ],
    pd.Series,
]

BacktestForecastFunction = Callable[
    [
        pd.DataFrame,
        dict[str, Any],
        pd.Series | None,
    ],
    pd.DataFrame,
]


def _require_mapping(
    config: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    """Return one required mapping from an ensemble YAML config."""

    value = config.get(key)

    if not isinstance(value, dict):
        raise TypeError(
            f"Ensemble config '{key}' must be a YAML mapping."
        )

    return value


def _get_ensemble_components(
    model_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Load and validate the ensemble component definitions."""

    components = _require_mapping(
        config=model_config,
        key="components",
    )

    if len(components) < 2:
        raise ValueError(
            "Ensemble requires at least two components."
        )

    normalized_components = {}

    for component_name, component_settings in (
        components.items()
    ):
        normalized_name = str(component_name)

        if not isinstance(component_settings, dict):
            raise TypeError(
                "Each ensemble component must be a YAML mapping: "
                f"{normalized_name}"
            )

        config_path = component_settings.get(
            "config_path",
        )

        if not config_path:
            raise ValueError(
                "Each ensemble component must contain config_path: "
                f"{normalized_name}"
            )

        if "weight" not in component_settings:
            raise ValueError(
                "Each ensemble component must contain weight: "
                f"{normalized_name}"
            )

        try:
            weight = float(
                component_settings["weight"]
            )
        except (
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                "Ensemble component weight must be numeric: "
                f"{normalized_name}"
            ) from error

        if weight <= 0:
            raise ValueError(
                "Ensemble component weight must be greater than zero: "
                f"{normalized_name}"
            )

        normalized_components[normalized_name] = {
            "config_path": Path(str(config_path)),
            "weight": weight,
        }

    total_weight = sum(
        component["weight"]
        for component in normalized_components.values()
    )

    if not np.isclose(
        total_weight,
        1.0,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "Ensemble component weights must sum to 1.0. "
            f"Current total: {total_weight}"
        )

    return normalized_components


def _load_component_config(
    component_name: str,
    config_path: Path,
    ensemble_run_id: str,
) -> dict[str, Any]:
    """Load one component YAML and reject invalid recursive structures."""

    if not config_path.exists():
        raise FileNotFoundError(
            "Cannot find ensemble component config: "
            f"{config_path}"
        )

    component_config = load_yaml(
        str(config_path)
    )

    if not isinstance(component_config, dict):
        raise ValueError(
            "Component config root must be a YAML mapping: "
            f"{config_path}"
        )

    model_settings = _require_mapping(
        config=component_config,
        key="model",
    )

    component_model_name = str(
        model_settings.get(
            "name",
            "",
        )
    ).lower()

    if component_model_name == "ensemble":
        raise ValueError(
            "Nested ensemble models are not allowed. "
            f"Component '{component_name}' points to: {config_path}"
        )

    component_run_id = str(
        model_settings.get(
            "run_id",
            "",
        )
    )

    if component_run_id == ensemble_run_id:
        raise ValueError(
            "Ensemble cannot include itself as a component: "
            f"{config_path}"
        )

    return component_config


def _validate_live_prediction(
    prediction: pd.Series,
    portfolio_tickers: list[str],
    component_name: str,
) -> pd.Series:
    """Validate one component's live prediction output."""

    if not isinstance(prediction, pd.Series):
        raise TypeError(
            "Ensemble live component must return a pandas Series: "
            f"{component_name}"
        )

    unexpected_tickers = [
        ticker
        for ticker in prediction.index
        if ticker not in portfolio_tickers
    ]

    if unexpected_tickers:
        raise ValueError(
            "Component prediction contains benchmark or unexpected "
            f"tickers: {unexpected_tickers}; "
            f"component={component_name}"
        )

    aligned_prediction = prediction.reindex(
        portfolio_tickers
    )

    if aligned_prediction.isna().any():
        missing_tickers = aligned_prediction[
            aligned_prediction.isna()
        ].index.tolist()

        raise ValueError(
            "Component prediction is missing portfolio tickers: "
            f"{missing_tickers}; "
            f"component={component_name}"
        )

    return aligned_prediction.astype(float)


def _validate_backtest_prediction(
    prediction: pd.DataFrame,
    portfolio_tickers: list[str],
    component_name: str,
) -> pd.DataFrame:
    """Validate one component's historical prediction table."""

    if not isinstance(prediction, pd.DataFrame):
        raise TypeError(
            "Ensemble backtest component must return a pandas DataFrame: "
            f"{component_name}"
        )

    unexpected_tickers = [
        ticker
        for ticker in prediction.columns
        if ticker not in portfolio_tickers
    ]

    if unexpected_tickers:
        raise ValueError(
            "Component backtest contains benchmark or unexpected "
            f"tickers: {unexpected_tickers}; "
            f"component={component_name}"
        )

    aligned_prediction = prediction.reindex(
        columns=portfolio_tickers
    )

    return aligned_prediction.astype(float)


def ensemble_live_forecast(
    weekly_returns: pd.DataFrame,
    model_config: dict[str, Any],
    benchmark_returns: pd.Series | None,
    child_forecast: LiveForecastFunction,
) -> pd.Series:
    """Return weighted next-week forecasts from all ensemble components."""

    model_settings = _require_mapping(
        config=model_config,
        key="model",
    )

    ensemble_run_id = str(
        model_settings.get(
            "run_id",
            "",
        )
    )

    if not ensemble_run_id:
        raise ValueError(
            "Ensemble model.run_id cannot be empty."
        )

    portfolio_tickers = list(
        weekly_returns.columns
    )

    components = _get_ensemble_components(
        model_config=model_config,
    )

    combined_prediction = pd.Series(
        0.0,
        index=portfolio_tickers,
        dtype=float,
        name="predicted_weekly_return",
    )

    for component_name, component_settings in (
        components.items()
    ):
        component_config = _load_component_config(
            component_name=component_name,
            config_path=component_settings[
                "config_path"
            ],
            ensemble_run_id=ensemble_run_id,
        )

        component_prediction = child_forecast(
            weekly_returns=weekly_returns,
            model_config=component_config,
            benchmark_returns=benchmark_returns,
        )

        component_prediction = _validate_live_prediction(
            prediction=component_prediction,
            portfolio_tickers=portfolio_tickers,
            component_name=component_name,
        )

        combined_prediction = (
            combined_prediction
            + component_settings["weight"]
            * component_prediction
        )

    return combined_prediction


def ensemble_backtest_forecast(
    weekly_returns: pd.DataFrame,
    model_config: dict[str, Any],
    benchmark_returns: pd.Series | None,
    child_backtest_forecast: BacktestForecastFunction,
) -> pd.DataFrame:
    """Return weighted historical forecasts from all ensemble components."""

    model_settings = _require_mapping(
        config=model_config,
        key="model",
    )

    ensemble_run_id = str(
        model_settings.get(
            "run_id",
            "",
        )
    )

    if not ensemble_run_id:
        raise ValueError(
            "Ensemble model.run_id cannot be empty."
        )

    portfolio_tickers = list(
        weekly_returns.columns
    )

    components = _get_ensemble_components(
        model_config=model_config,
    )

    combined_prediction = pd.DataFrame(
        0.0,
        index=weekly_returns.index,
        columns=portfolio_tickers,
        dtype=float,
    )

    component_validity = pd.DataFrame(
        True,
        index=weekly_returns.index,
        columns=portfolio_tickers,
    )

    for component_name, component_settings in (
        components.items()
    ):
        component_config = _load_component_config(
            component_name=component_name,
            config_path=component_settings[
                "config_path"
            ],
            ensemble_run_id=ensemble_run_id,
        )

        component_prediction = child_backtest_forecast(
            weekly_returns=weekly_returns,
            model_config=component_config,
            benchmark_returns=benchmark_returns,
        )

        component_prediction = _validate_backtest_prediction(
            prediction=component_prediction,
            portfolio_tickers=portfolio_tickers,
            component_name=component_name,
        )

        component_prediction = component_prediction.reindex(
            index=weekly_returns.index,
            columns=portfolio_tickers,
        )

        component_validity = (
            component_validity
            & component_prediction.notna()
        )

        combined_prediction = (
            combined_prediction
            + component_settings["weight"]
            * component_prediction.fillna(0.0)
        )

    combined_prediction = combined_prediction.where(
        component_validity
    )

    return combined_prediction