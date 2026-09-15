"""Validation and trial helpers for regression Optuna tuning studies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.hk_equity.data.preprocess import (
    calculate_returns,
    to_weekly_close,
)
from src.hk_equity.models.registry import (
    get_backtest_forecasts,
)
from src.hk_equity.tuning.objectives import (
    get_objective_spec,
)
from src.hk_equity.utils.config import (
    load_yaml,
)


REQUIRED_TOP_LEVEL_KEYS = {
    "study",
    "target",
    "evaluation",
    "training",
    "features",
    "search_space",
    "fixed_parameters",
    "output",
}


REQUIRED_REGRESSION_SEARCH_SPACE_KEYS = {
    "estimator",
    "ridge_alpha",
    "lasso_alpha",
    "elastic_net_alpha",
    "elastic_net_l1_ratio",
}


VALID_ESTIMATORS = {
    "ridge",
    "lasso",
    "elastic_net",
}


def _require_mapping(
    config: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    """Return one required YAML mapping."""

    value = config.get(key)

    if not isinstance(value, dict):
        raise TypeError(
            f"Tuning config '{key}' must be a YAML mapping."
        )

    return value


def _require_list(
    config: dict[str, Any],
    key: str,
) -> list[Any]:
    """Return one required YAML list."""

    value = config.get(key)

    if not isinstance(value, list):
        raise TypeError(
            f"Tuning config '{key}' must be a YAML list."
        )

    return value


def _validate_unique_strings(
    values: list[Any],
    name: str,
) -> list[str]:
    """Convert a YAML list to unique strings."""

    normalized_values = [
        str(value)
        for value in values
    ]

    if len(normalized_values) != len(
        set(normalized_values)
    ):
        raise ValueError(
            f"{name} contains duplicate values."
        )

    return normalized_values


def _validate_positive_integer(
    value: Any,
    name: str,
) -> int:
    """Return one strictly positive integer."""

    try:
        integer_value = int(value)
    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            f"{name} must be a positive integer."
        ) from error

    if integer_value <= 0:
        raise ValueError(
            f"{name} must be greater than zero."
        )

    return integer_value


def _validate_evaluation_period(
    evaluation: dict[str, Any],
) -> None:
    """Validate tuning evaluation start and end dates."""

    if "start" not in evaluation or "end" not in evaluation:
        raise ValueError(
            "evaluation must contain start and end."
        )

    start = pd.Timestamp(
        evaluation["start"]
    )

    end = pd.Timestamp(
        evaluation["end"]
    )

    if start > end:
        raise ValueError(
            "evaluation.start must not be later than "
            "evaluation.end."
        )


def _validate_numeric_search_space(
    parameter_space: dict[str, Any],
    name: str,
) -> None:
    """Validate one numeric Optuna search-space mapping."""

    if (
        "low" not in parameter_space
        or "high" not in parameter_space
    ):
        raise ValueError(
            f"search_space.{name} must contain low and high."
        )

    low = float(parameter_space["low"])
    high = float(parameter_space["high"])

    if low <= 0 or high <= 0:
        raise ValueError(
            f"search_space.{name} bounds must be greater than zero."
        )

    if low > high:
        raise ValueError(
            f"search_space.{name}.low must not exceed high."
        )


def _validate_search_space(
    search_space: dict[str, Any],
) -> None:
    """Validate all required regression search spaces."""

    missing_keys = (
        REQUIRED_REGRESSION_SEARCH_SPACE_KEYS
        - set(search_space)
    )

    if missing_keys:
        raise ValueError(
            "search_space is missing required regression "
            f"parameters: {sorted(missing_keys)}"
        )

    estimators = _require_list(
        config=search_space,
        key="estimator",
    )

    estimators = _validate_unique_strings(
        values=estimators,
        name="search_space.estimator",
    )

    normalized_estimators = {
        estimator.lower()
        for estimator in estimators
    }

    if not normalized_estimators:
        raise ValueError(
            "search_space.estimator cannot be empty."
        )

    invalid_estimators = (
        normalized_estimators
        - VALID_ESTIMATORS
    )

    if invalid_estimators:
        raise ValueError(
            "search_space.estimator contains invalid values: "
            f"{sorted(invalid_estimators)}"
        )

    for parameter_name in {
        "ridge_alpha",
        "lasso_alpha",
        "elastic_net_alpha",
        "elastic_net_l1_ratio",
    }:
        parameter_space = search_space[
            parameter_name
        ]

        if not isinstance(parameter_space, dict):
            raise TypeError(
                "Each numeric search_space parameter must be "
                f"a YAML mapping: {parameter_name}"
            )

        _validate_numeric_search_space(
            parameter_space=parameter_space,
            name=parameter_name,
        )

    l1_ratio_space = search_space[
        "elastic_net_l1_ratio"
    ]

    if not (
        0.0 <= float(l1_ratio_space["low"])
        <= float(l1_ratio_space["high"])
        <= 1.0
    ):
        raise ValueError(
            "search_space.elastic_net_l1_ratio must be "
            "between zero and one."
        )


def load_regression_tuning_config(
    config_path: str | Path,
    portfolio_tickers: list[str],
) -> dict[str, Any]:
    """Load and validate one regression Optuna tuning configuration."""

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Cannot find tuning config: {config_path}"
        )

    config = load_yaml(config_path)

    missing_top_level_keys = (
        REQUIRED_TOP_LEVEL_KEYS
        - set(config)
    )

    if missing_top_level_keys:
        raise ValueError(
            "Tuning config is missing top-level keys: "
            f"{sorted(missing_top_level_keys)}"
        )

    study = _require_mapping(config, "study")
    target = _require_mapping(config, "target")
    evaluation = _require_mapping(config, "evaluation")
    training = _require_mapping(config, "training")
    features = _require_mapping(config, "features")
    search_space = _require_mapping(config, "search_space")
    fixed_parameters = _require_mapping(
        config,
        "fixed_parameters",
    )
    output = _require_mapping(config, "output")

    study_name = str(study.get("name", ""))

    if not study_name:
        raise ValueError("study.name cannot be empty.")

    objective_name = str(study.get("objective", ""))
    objective_spec = get_objective_spec(
        objective_name=objective_name,
    )

    configured_direction = str(
        study.get("direction", "")
    ).lower()

    if configured_direction != objective_spec.direction:
        raise ValueError(
            "study.direction does not match the objective. "
            f"Expected '{objective_spec.direction}' for "
            f"objective '{objective_spec.name}'."
        )

    _validate_positive_integer(
        value=study.get("n_trials"),
        name="study.n_trials",
    )

    _validate_positive_integer(
        value=training.get("minimum_training_weeks"),
        name="training.minimum_training_weeks",
    )

    _validate_positive_integer(
        value=fixed_parameters.get("max_iter"),
        name="fixed_parameters.max_iter",
    )

    _validate_evaluation_period(evaluation)

    if not isinstance(
        fixed_parameters.get("standardize"),
        bool,
    ):
        raise TypeError(
            "fixed_parameters.standardize must be Boolean."
        )

    if not isinstance(
        fixed_parameters.get("fit_intercept"),
        bool,
    ):
        raise TypeError(
            "fixed_parameters.fit_intercept must be Boolean."
        )

    valid_portfolio_tickers = {
        str(ticker)
        for ticker in portfolio_tickers
    }

    base_ticker = str(target.get("base_ticker", ""))

    if base_ticker not in valid_portfolio_tickers:
        raise ValueError(
            "target.base_ticker must be one of the "
            "portfolio tickers. Invalid value: "
            f"'{base_ticker}'"
        )

    peer_candidates = _require_list(
        config=training,
        key="peer_candidates",
    )

    peer_candidates = _validate_unique_strings(
        values=peer_candidates,
        name="training.peer_candidates",
    )

    invalid_peer_tickers = [
        ticker
        for ticker in peer_candidates
        if ticker not in valid_portfolio_tickers
    ]

    if invalid_peer_tickers:
        raise ValueError(
            "training.peer_candidates contains invalid "
            "portfolio tickers: "
            f"{invalid_peer_tickers}"
        )

    if base_ticker in peer_candidates:
        raise ValueError(
            "training.peer_candidates must not contain "
            "target.base_ticker because the base stock "
            "is included automatically."
        )

    min_total_tickers = _validate_positive_integer(
        value=training.get("min_total_tickers"),
        name="training.min_total_tickers",
    )

    max_total_tickers = _validate_positive_integer(
        value=training.get("max_total_tickers"),
        name="training.max_total_tickers",
    )

    if min_total_tickers > max_total_tickers:
        raise ValueError(
            "training.min_total_tickers must not be "
            "greater than training.max_total_tickers."
        )

    maximum_available_tickers = len(peer_candidates) + 1

    if max_total_tickers > maximum_available_tickers:
        raise ValueError(
            "training.max_total_tickers exceeds the base "
            "stock plus available peer candidates."
        )

    candidate_columns = _require_list(
        config=features,
        key="candidate_columns",
    )

    candidate_columns = _validate_unique_strings(
        values=candidate_columns,
        name="features.candidate_columns",
    )

    if not candidate_columns:
        raise ValueError(
            "features.candidate_columns cannot be empty."
        )

    min_features = _validate_positive_integer(
        value=features.get("min_features"),
        name="features.min_features",
    )

    max_features = _validate_positive_integer(
        value=features.get("max_features"),
        name="features.max_features",
    )

    if min_features > max_features:
        raise ValueError(
            "features.min_features must not be greater "
            "than features.max_features."
        )

    if max_features > len(candidate_columns):
        raise ValueError(
            "features.max_features exceeds the number of "
            "features.candidate_columns."
        )

    _validate_search_space(search_space)

    output_run_id = str(output.get("run_id", ""))

    if not output_run_id:
        raise ValueError("output.run_id cannot be empty.")

    output_directory = str(
        output.get("output_directory", "")
    )

    if not output_directory:
        raise ValueError(
            "output.output_directory cannot be empty."
        )

    return config


def build_trial_model_config(
    tuning_config: dict[str, Any],
    trial,
) -> dict[str, Any]:
    """Build one temporary regression model config from an Optuna trial."""

    try:
        import optuna
    except ImportError as error:
        raise ImportError(
            "Optuna is required to build a tuning trial."
        ) from error

    target_settings = tuning_config["target"]
    training_settings = tuning_config["training"]
    feature_settings = tuning_config["features"]
    search_space = tuning_config["search_space"]
    fixed_parameters = tuning_config["fixed_parameters"]

    base_ticker = str(target_settings["base_ticker"])

    selected_peers = [
        ticker
        for ticker in training_settings["peer_candidates"]
        if trial.suggest_categorical(
            f"use_peer_{ticker}",
            [True, False],
        )
    ]

    training_tickers = [base_ticker, *selected_peers]

    if not (
        int(training_settings["min_total_tickers"])
        <= len(training_tickers)
        <= int(training_settings["max_total_tickers"])
    ):
        raise optuna.TrialPruned(
            "Selected training ticker count is outside "
            "the configured range."
        )

    selected_columns = [
        column
        for column in feature_settings["candidate_columns"]
        if trial.suggest_categorical(
            f"use_feature_{column}",
            [True, False],
        )
    ]

    if not (
        int(feature_settings["min_features"])
        <= len(selected_columns)
        <= int(feature_settings["max_features"])
    ):
        raise optuna.TrialPruned(
            "Selected feature count is outside the "
            "configured range."
        )

    estimator = trial.suggest_categorical(
        "estimator",
        search_space["estimator"],
    ).lower()

    if estimator == "ridge":
        alpha_space = search_space["ridge_alpha"]
        alpha = trial.suggest_float(
            "ridge_alpha",
            float(alpha_space["low"]),
            float(alpha_space["high"]),
            log=bool(alpha_space.get("log", False)),
        )
        regression_parameters = {
            "estimator": "ridge",
            "alpha": alpha,
        }

    elif estimator == "lasso":
        alpha_space = search_space["lasso_alpha"]
        alpha = trial.suggest_float(
            "lasso_alpha",
            float(alpha_space["low"]),
            float(alpha_space["high"]),
            log=bool(alpha_space.get("log", False)),
        )
        regression_parameters = {
            "estimator": "lasso",
            "alpha": alpha,
        }

    else:
        alpha_space = search_space["elastic_net_alpha"]
        l1_ratio_space = search_space[
            "elastic_net_l1_ratio"
        ]

        alpha = trial.suggest_float(
            "elastic_net_alpha",
            float(alpha_space["low"]),
            float(alpha_space["high"]),
            log=bool(alpha_space.get("log", False)),
        )

        l1_ratio = trial.suggest_float(
            "elastic_net_l1_ratio",
            float(l1_ratio_space["low"]),
            float(l1_ratio_space["high"]),
        )

        regression_parameters = {
            "estimator": "elastic_net",
            "alpha": alpha,
            "l1_ratio": l1_ratio,
        }

    return {
        "model": {
            "name": "general_regression",
            "label": (
                f"{base_ticker} Regression Optuna Trial"
            ),
            "version": "trial",
            "run_id": (
                f"{tuning_config['output']['run_id']}"
                f"_trial_{trial.number}"
            ),
        },
        "parameters": {
            **fixed_parameters,
            **regression_parameters,
            "minimum_training_weeks": int(
                training_settings["minimum_training_weeks"]
            ),
        },
        "training": {
            "tickers": training_tickers,
        },
        "features": {
            "feature_set": feature_settings["feature_set"],
            "feature_map_path": feature_settings[
                "feature_map_path"
            ],
            "selected_columns": selected_columns,
        },
    }


def prepare_tuning_weekly_returns(
    base_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    """Build portfolio and benchmark weekly returns for one tuning study."""

    raw_data_directory = Path(
        base_config["paths"]["raw_data"]
    )

    daily_close_path = (
        raw_data_directory
        / "latest_daily_close.csv"
    )

    if not daily_close_path.exists():
        raise FileNotFoundError(
            f"Cannot find: {daily_close_path}"
        )

    daily_close = pd.read_csv(
        daily_close_path,
        index_col="Date",
        parse_dates=True,
    ).sort_index()

    if daily_close.empty:
        raise ValueError(
            "latest_daily_close.csv is empty."
        )

    portfolio_tickers = list(
        base_config["tickers"].keys()
    )

    benchmark_ticker = str(
        base_config["market"]["ticker"]
    )

    missing_portfolio_tickers = [
        ticker
        for ticker in portfolio_tickers
        if ticker not in daily_close.columns
    ]

    if missing_portfolio_tickers:
        raise ValueError(
            "Missing portfolio ticker columns: "
            f"{missing_portfolio_tickers}"
        )

    if benchmark_ticker not in daily_close.columns:
        raise ValueError(
            f"Missing benchmark ticker column: {benchmark_ticker}"
        )

    weekly_frequency = base_config["forecast"][
        "weekly_frequency"
    ]

    portfolio_weekly_close = to_weekly_close(
        daily_close=daily_close[portfolio_tickers].copy(),
        weekly_frequency=weekly_frequency,
    )

    benchmark_weekly_close = to_weekly_close(
        daily_close=daily_close[benchmark_ticker]
        .dropna()
        .to_frame(),
        weekly_frequency=weekly_frequency,
    )

    weekly_returns = calculate_returns(
        portfolio_weekly_close
    ).dropna(how="all")

    benchmark_returns = (
        calculate_returns(benchmark_weekly_close)
        .iloc[:, 0]
        .dropna()
    )

    benchmark_returns.name = benchmark_ticker

    if weekly_returns.empty:
        raise ValueError(
            "No portfolio weekly returns could be calculated."
        )

    if benchmark_returns.empty:
        raise ValueError(
            "No benchmark weekly returns could be calculated."
        )

    return weekly_returns, benchmark_returns


def evaluate_regression_trial(
    trial,
    tuning_config: dict[str, Any],
    weekly_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> float:
    """Evaluate one regression Optuna trial for one base-stock objective."""

    try:
        import optuna
    except ImportError as error:
        raise ImportError(
            "Optuna is required to evaluate a tuning trial."
        ) from error

    trial_model_config = build_trial_model_config(
        tuning_config=tuning_config,
        trial=trial,
    )

    base_ticker = str(
        tuning_config["target"]["base_ticker"]
    )

    evaluation_start = pd.Timestamp(
        tuning_config["evaluation"]["start"]
    )

    evaluation_end = pd.Timestamp(
        tuning_config["evaluation"]["end"]
    )

    # Do not compute post-evaluation forecasts during tuning.
    weekly_returns = weekly_returns.loc[:evaluation_end]
    benchmark_returns = benchmark_returns.loc[:evaluation_end]

    predicted_returns = get_backtest_forecasts(
        weekly_returns=weekly_returns,
        benchmark_returns=benchmark_returns,
        model_config=trial_model_config,
    )

    actual_base_returns = weekly_returns.loc[
        evaluation_start:evaluation_end,
        base_ticker,
    ]

    predicted_base_returns = predicted_returns.loc[
        evaluation_start:evaluation_end,
        base_ticker,
    ]

    objective_spec = get_objective_spec(
        tuning_config["study"]["objective"]
    )

    objective_settings = tuning_config["study"].get(
        "objective_settings",
        {},
    )

    try:
        score = objective_spec.score_function(
            actual=actual_base_returns,
            predicted=predicted_base_returns,
            settings=objective_settings,
        )
    except ValueError as error:
        raise optuna.TrialPruned(
            f"Trial cannot be scored: {error}"
        ) from error

    trial.set_user_attr("base_ticker", base_ticker)

    trial.set_user_attr(
        "training_tickers",
        trial_model_config["training"]["tickers"],
    )

    trial.set_user_attr(
        "selected_columns",
        trial_model_config["features"]["selected_columns"],
    )

    trial.set_user_attr(
        "regression_parameters",
        trial_model_config["parameters"],
    )

    trial.set_user_attr(
        "model_config",
        trial_model_config,
    )

    return float(score)

