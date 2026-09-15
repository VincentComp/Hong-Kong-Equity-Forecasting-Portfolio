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
OUTPUT_DIRECTORY = Path("configs/tuning/generated")

FEATURE_SET = "weekly_v1"
FEATURE_MAP_PATH = "data/processed/weekly_features_weekly_v1.csv"

MINIMUM_TRAINING_WEEKS = 104
MIN_FEATURES = 5
MAX_FEATURES = 20
DEFAULT_N_TRIALS = 200
SAMPLER_SEED = 42


# =============================================================================
# Version definitions
# =============================================================================

VERSION_PERIODS = {
    "v1": {
        "start": "2025-01-01",
        "end": "2025-12-31",
        "description": "2025-only validation period.",
    },
    "v2": {
        "start": "2022-01-01",
        "end": "2025-12-31",
        "description": (
            "Long-history validation period. Early weeks without "
            "minimum_training_weeks are skipped by the objective."
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
        "low": 0.001,
        "high": 100.0,
        "log": True,
    },
    "lasso_alpha": {
        "low": 0.00001,
        "high": 0.1,
        "log": True,
    },
    "elastic_net_alpha": {
        "low": 0.00001,
        "high": 0.1,
        "log": True,
    },
    "elastic_net_l1_ratio": {
        "low": 0.05,
        "high": 0.95,
    },
}

REGRESSION_FIXED_PARAMETERS = {
    "standardize": True,
    "fit_intercept": True,
    "max_iter": 10000,
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
        f"{ticker_id}_{model_family}_mae_optuna_{version}"
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

    if model_family not in {
        "xgboost",
        "regression",
    }:
        raise ValueError(
            f"Unknown model family: {model_family}"
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

    if model_family == "xgboost":
        search_space = XGBOOST_SEARCH_SPACE
        fixed_parameters = XGBOOST_FIXED_PARAMETERS

    else:
        search_space = REGRESSION_SEARCH_SPACE
        fixed_parameters = REGRESSION_FIXED_PARAMETERS

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
            "note": (
                "HSI (^HSI) is excluded from base_ticker and "
                "peer_candidates. It remains benchmark-only."
            ),
        },
    }


def write_tuning_config(
    config: dict[str, Any],
    output_path: Path,
) -> None:
    """Save one YAML config without overwriting an existing file."""

    if output_path.exists():
        raise FileExistsError(
            "Refusing to overwrite existing config: "
            f"{output_path}"
        )

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
    """Create all 40 ticker-specific Optuna tuning YAML files."""

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

    generated_paths = []

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

                study_name = config["study"]["name"]

                output_path = (
                    OUTPUT_DIRECTORY
                    / f"{study_name}.yaml"
                )

                write_tuning_config(
                    config=config,
                    output_path=output_path,
                )

                generated_paths.append(
                    output_path
                )

    print(
        "\nGenerated tuning configurations:"
    )

    for path in generated_paths:
        print(path)

    print(
        f"\nTotal generated: {len(generated_paths)}"
    )

    print(
        "\nVersion definitions:"
        "\nv1 = validation: 2025-01-01 to 2025-12-31"
        "\nv2 = validation: 2022-01-01 to 2025-12-31"
        "\n     (early weeks are skipped until "
        "minimum_training_weeks is satisfied)"
    )


if __name__ == "__main__":
    main()