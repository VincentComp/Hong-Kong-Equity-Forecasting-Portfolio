"""Create ticker-specific Optuna tuning configs for XGBoost and regression.

Creates:
    10 portfolio stocks x 2 model families x 2 evaluation versions
    = 40 YAML files.

Version definition:
    v1: validation period 2025-01-01 to 2025-12-31
    v2: validation period 2022-01-01 to 2025-12-31

The Hang Seng Index (^HSI) is never a target or peer candidate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.hk_equity.utils.config import load_yaml


# =============================================================================
# Shared project settings
# =============================================================================

BASE_CONFIG_PATH = "configs/base.yaml"

# Keep this separate from earlier generated configs and existing Optuna studies.
OUTPUT_DIRECTORY = Path("configs/tuning/generated_stable")

FEATURE_SET = "weekly_v1"
FEATURE_MAP_PATH = "data/processed/weekly_features_weekly_v1.csv"

MINIMUM_TRAINING_WEEKS = 104
MIN_FEATURES = 5
MAX_FEATURES = 20
DEFAULT_N_TRIALS = 200
SAMPLER_SEED = 42

# Makes this batch independent from any earlier studies and SQLite databases.
STUDY_SUFFIX = "stable"


# =============================================================================
# Version definitions
# =============================================================================

VERSION_PERIODS = {
    "v1": {
        "start": "2025-01-01",
        "end": "2025-12-31",
        "description": (
            "Recent-regime validation period from 2025-01-01 "
            "to 2025-12-31."
        ),
    },
    "v2": {
        "start": "2022-01-01",
        "end": "2025-12-31",
        "description": (
            "Long-history validation period from 2022-01-01 "
            "to 2025-12-31. Each expanding-window forecast uses "
            "only historical observations before its target week."
        ),
    },
}


# =============================================================================
# Weekly V1 feature candidates
# =============================================================================

CANDIDATE_COLUMNS = [
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_4",
    "return_lag_8",
    "return_lag_12",
    "return_lag_24",
    "return_lag_52",
    "rolling_mean_4",
    "rolling_std_4",
    "downside_deviation_4",
    "risk_adjusted_mean_4",
    "rolling_mean_12",
    "rolling_std_12",
    "downside_deviation_12",
    "risk_adjusted_mean_12",
    "rolling_mean_26",
    "rolling_std_26",
    "downside_deviation_26",
    "risk_adjusted_mean_26",
    "momentum_spread_4_12",
    "market_return_lag_1",
    "market_rolling_mean_4",
    "market_rolling_mean_12",
    "market_rolling_mean_26",
    "excess_return_lag_1",
    "excess_return_lag_2",
    "excess_return_lag_3",
    "excess_return_lag_4",
    "excess_return_lag_8",
    "excess_return_lag_12",
    "excess_return_lag_24",
    "excess_return_lag_52",
    "excess_rolling_mean_4",
    "excess_rolling_mean_12",
    "excess_rolling_mean_26",
    "return_lag_1_cs_z",
    "return_lag_4_cs_z",
    "return_lag_12_cs_z",
    "rolling_mean_4_cs_z",
    "rolling_mean_12_cs_z",
    "rolling_mean_26_cs_z",
    "risk_adjusted_mean_4_cs_z",
    "excess_return_lag_1_cs_z",
]


# =============================================================================
# Search spaces
# =============================================================================

XGBOOST_SEARCH_SPACE = {
    "n_estimators": {
        "low": 40,
        "high": 120,
        "step": 10,
    },
    "max_depth": {
        "low": 2,
        "high": 4,
    },
    "learning_rate": {
        "low": 0.005,
        "high": 0.04,
        "log": True,
    },
    "subsample": {
        "low": 0.6,
        "high": 0.9,
    },
    "colsample_bytree": {
        "low": 0.6,
        "high": 0.9,
    },
    "reg_alpha": {
        "low": 0.001,
        "high": 0.2,
        "log": True,
    },
    "reg_lambda": {
        "low": 0.001,
        "high": 0.2,
        "log": True,
    },
    "min_child_weight": {
        "low": 2,
        "high": 8,
    },
}

XGBOOST_FIXED_PARAMETERS = {
    "estimator": "xgboost",
    "random_state": 42,
    "n_jobs": -1,
}

REGRESSION_SEARCH_SPACE = {
    "estimator": [
        "ridge",
        "lasso",
        "elastic_net",
    ],
    "ridge_alpha": {
        "low": 0.01,
        "high": 100.0,
        "log": True,
    },
    "lasso_alpha": {
        "low": 0.0001,
        "high": 0.1,
        "log": True,
    },
    "elastic_net_alpha": {
        "low": 0.0001,
        "high": 0.1,
        "log": True,
    },
    "elastic_net_l1_ratio": {
        "low": 0.10,
        "high": 0.90,
    },
}

REGRESSION_FIXED_PARAMETERS = {
    "standardize": True,
    "fit_intercept": True,
    "max_iter": 30000,
    "tol": 0.0001,
}


# =============================================================================
# YAML construction helpers
# =============================================================================

def get_study_name(
    ticker: str,
    model_family: str,
    version: str,
) -> str:
    """Return a stable, filesystem-safe study identifier."""

    ticker_id = ticker.replace(".HK", "").lower()

    return (
        f"{ticker_id}_{model_family}_mae_optuna_"
        f"{version}_{STUDY_SUFFIX}"
    )


def get_output_settings(
    study_name: str,
) -> dict[str, Any]:
    """Create isolated Optuna artifact paths for one study."""

    output_directory = f"outputs/tuning/{study_name}"

    return {
        "run_id": study_name,
        "output_directory": output_directory,
        "storage_file": (
            f"{output_directory}/optuna_studies.db"
        ),
        "load_if_exists": True,
        "trials_csv": "trials.csv",
        "best_trial_json": "best_trial.json",
        "best_model_config_yaml": "best_model_config.yaml",
        "tuning_metadata_json": "tuning_metadata.json",
    }


def get_model_settings(
    model_family: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the search space and fixed parameters for one model family."""

    if model_family == "xgboost":
        return (
            XGBOOST_SEARCH_SPACE,
            XGBOOST_FIXED_PARAMETERS,
        )

    if model_family == "regression":
        return (
            REGRESSION_SEARCH_SPACE,
            REGRESSION_FIXED_PARAMETERS,
        )

    raise ValueError(
        f"Unknown model family: {model_family}"
    )


def build_tuning_config(
    ticker: str,
    ticker_name: str,
    portfolio_tickers: list[str],
    model_family: str,
    version: str,
) -> dict[str, Any]:
    """Build one Optuna tuning configuration dictionary."""

    if ticker not in portfolio_tickers:
        raise ValueError(
            f"Base ticker is not in portfolio: {ticker}"
        )

    if version not in VERSION_PERIODS:
        raise ValueError(
            f"Unknown version: {version}"
        )

    peer_candidates = [
        peer_ticker
        for peer_ticker in portfolio_tickers
        if peer_ticker != ticker
    ]

    period = VERSION_PERIODS[version]

    study_name = get_study_name(
        ticker=ticker,
        model_family=model_family,
        version=version,
    )

    search_space, fixed_parameters = (
        get_model_settings(
            model_family=model_family,
        )
    )

    return {
        "tuning": {
            "model_family": model_family,
        },
        "study": {
            "name": study_name,
            "objective": "mae",
            "direction": "minimize",
            "n_trials": DEFAULT_N_TRIALS,
            "sampler_seed": SAMPLER_SEED,
        },
        "target": {
            "base_ticker": ticker,
        },
        "evaluation": {
            "start": period["start"],
            "end": period["end"],
        },
        "training": {
            "minimum_training_weeks": (
                MINIMUM_TRAINING_WEEKS
            ),
            "peer_candidates": peer_candidates,
            "min_total_tickers": 1,
            "max_total_tickers": len(portfolio_tickers),
        },
        "features": {
            "feature_set": FEATURE_SET,
            "feature_map_path": FEATURE_MAP_PATH,
            "candidate_columns": CANDIDATE_COLUMNS,
            "min_features": MIN_FEATURES,
            "max_features": MAX_FEATURES,
        },
        "search_space": search_space,
        "fixed_parameters": fixed_parameters,
        "output": get_output_settings(
            study_name=study_name,
        ),
        "metadata": {
            "base_ticker_name": ticker_name,
            "tuning_version": version,
            "version_description": period["description"],
            "study_suffix": STUDY_SUFFIX,
            "note": (
                "HSI (^HSI) is excluded from base_ticker and "
                "peer_candidates. It remains benchmark-only."
            ),
        },
    }


def validate_tuning_config(
    config: dict[str, Any],
    benchmark_ticker: str,
) -> None:
    """Validate one generated configuration before it is written."""

    required_sections = {
        "tuning",
        "study",
        "target",
        "evaluation",
        "training",
        "features",
        "search_space",
        "fixed_parameters",
        "output",
        "metadata",
    }

    missing_sections = (
        required_sections - set(config.keys())
    )

    if missing_sections:
        raise KeyError(
            "Generated config is missing sections: "
            f"{sorted(missing_sections)}"
        )

    model_family = config["tuning"]["model_family"]

    if model_family not in {
        "xgboost",
        "regression",
    }:
        raise ValueError(
            f"Invalid model family: {model_family}"
        )

    base_ticker = config["target"]["base_ticker"]

    peer_candidates = config["training"][
        "peer_candidates"
    ]

    if base_ticker == benchmark_ticker:
        raise ValueError(
            "Benchmark cannot be the base ticker: "
            f"{benchmark_ticker}"
        )

    if benchmark_ticker in peer_candidates:
        raise ValueError(
            "Benchmark cannot be a peer candidate: "
            f"{benchmark_ticker}"
        )

    if base_ticker in peer_candidates:
        raise ValueError(
            "Base ticker must not be in peer_candidates: "
            f"{base_ticker}"
        )

    if len(peer_candidates) != len(
        set(peer_candidates)
    ):
        raise ValueError(
            "peer_candidates contains duplicates: "
            f"{base_ticker}"
        )

    min_features = config["features"][
        "min_features"
    ]

    max_features = config["features"][
        "max_features"
    ]

    candidate_columns = config["features"][
        "candidate_columns"
    ]

    if min_features <= 0:
        raise ValueError(
            "min_features must be greater than zero."
        )

    if max_features < min_features:
        raise ValueError(
            "max_features must be at least min_features."
        )

    if max_features > len(candidate_columns):
        raise ValueError(
            "max_features cannot exceed number of "
            "candidate columns."
        )

    evaluation_start = config["evaluation"]["start"]
    evaluation_end = config["evaluation"]["end"]

    if evaluation_start >= evaluation_end:
        raise ValueError(
            "evaluation.start must be earlier than "
            "evaluation.end."
        )

    output = config["output"]

    required_output_keys = {
        "run_id",
        "output_directory",
        "storage_file",
    }

    missing_output_keys = (
        required_output_keys - set(output.keys())
    )

    if missing_output_keys:
        raise KeyError(
            "output is missing keys: "
            f"{sorted(missing_output_keys)}"
        )


def validate_output_paths(
    output_paths: list[Path],
) -> None:
    """Fail before writing when any target config already exists."""

    existing_paths = [
        path
        for path in output_paths
        if path.exists()
    ]

    if existing_paths:
        formatted_paths = "\n".join(
            str(path)
            for path in existing_paths
        )

        raise FileExistsError(
            "Refusing to overwrite existing generated configs:\n"
            f"{formatted_paths}"
        )


def write_tuning_config(
    config: dict[str, Any],
    output_path: Path,
) -> None:
    """Save one validated YAML config."""

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config,
            file,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )


def main() -> None:
    """Create all 40 stable ticker-specific Optuna tuning YAML files."""

    base_config = load_yaml(
        BASE_CONFIG_PATH
    )

    portfolio = base_config["tickers"]
    portfolio_tickers = list(portfolio.keys())

    benchmark_ticker = base_config["market"]["ticker"]

    if benchmark_ticker in portfolio_tickers:
        raise ValueError(
            "Benchmark must not be a portfolio ticker: "
            f"{benchmark_ticker}"
        )

    if len(portfolio_tickers) != 10:
        raise ValueError(
            "Expected exactly 10 portfolio stocks, found: "
            f"{len(portfolio_tickers)}"
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_records = []

    for model_family in [
        "xgboost",
        "regression",
    ]:
        for version in [
            "v1",
            "v2",
        ]:
            for ticker, ticker_name in portfolio.items():
                config = build_tuning_config(
                    ticker=ticker,
                    ticker_name=ticker_name,
                    portfolio_tickers=portfolio_tickers,
                    model_family=model_family,
                    version=version,
                )

                validate_tuning_config(
                    config=config,
                    benchmark_ticker=benchmark_ticker,
                )

                study_name = config["study"]["name"]

                output_path = (
                    OUTPUT_DIRECTORY
                    / f"{study_name}.yaml"
                )

                config_records.append(
                    (output_path, config)
                )

    output_paths = [
        output_path
        for output_path, _ in config_records
    ]

    validate_output_paths(
        output_paths=output_paths,
    )

    for output_path, config in config_records:
        write_tuning_config(
            config=config,
            output_path=output_path,
        )

    print(
        "\nGenerated stable tuning configurations:"
    )

    for output_path, _ in config_records:
        print(output_path)

    print(
        f"\nTotal generated: {len(config_records)}"
    )

    print(
        "\nVersion definitions:"
        "\nv1 = validation: 2025-01-01 to 2025-12-31"
        "\nv2 = validation: 2022-01-01 to 2025-12-31"
        "\n     (each forecast trains only on earlier data)"
    )

    print(
        "\nRegression refinement:"
        "\n- Lasso alpha range: 0.0001 to 0.1"
        "\n- Elastic Net alpha range: 0.0001 to 0.1"
        "\n- Elastic Net l1_ratio range: 0.10 to 0.90"
        "\n- max_iter: 30000"
        "\n- tol: 0.0001"
    )


if __name__ == "__main__":
    main()