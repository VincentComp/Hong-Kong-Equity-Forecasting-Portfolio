"""Temporarily run portfolio-wide Optuna tuning for regression or XGBoost.

This script is deliberately independent from the existing ticker-specific
Optuna modules. It tunes one global model using all 10 portfolio stocks:

- Training universe: all portfolio stocks in configs/base.yaml
- Validation objective: equal-weight mean MAE across all portfolio stocks
- Evaluation period: 2025-01-01 to 2025-12-31
- HSI: benchmark-only; never a portfolio target
- Optuna tuning: hyperparameters only
- Features: fixed selected_columns supplied by the command line configuration

Examples:
    python -m scripts.run_portfolio_optuna \ # <---- Change path
        --model-family xgboost \
        --n-trials 3

    python -m scripts.run_portfolio_optuna \
        --model-family xgboost \
        --n-trials 100

    python -m scripts.run_portfolio_optuna \
        --model-family regression \
        --n-trials 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from xgboost import XGBRegressor

from src.hk_equity.data.preprocess import (
    calculate_returns,
    to_weekly_close,
)
from src.hk_equity.features.feature_config import (
    build_weekly_feature_spec,
    load_feature_set_config,
)
from src.hk_equity.features.weekly_features import (
    build_weekly_training_features,
    get_feature_columns,
)
from src.hk_equity.utils.config import (
    load_yaml,
)


DEFAULT_BASE_CONFIG = "configs/base.yaml"
DEFAULT_FEATURE_SET = "weekly_v1"
DEFAULT_FEATURE_MAP = (
    "data/processed/weekly_features_weekly_v1.csv"
)
DEFAULT_VALIDATION_START = "2025-01-01"
DEFAULT_VALIDATION_END = "2025-12-31"
DEFAULT_MINIMUM_TRAINING_WEEKS = 104
DEFAULT_N_TRIALS = 100
DEFAULT_SEED = 42

VALIDATION_PERIODS = {
    "v1": {
        "start": "2025-01-01",
        "end": "2025-12-31",
    },
    "v2": {
        "start": "2022-01-01",
        "end": "2025-12-31",
    },
}


def parse_arguments() -> argparse.Namespace:
    """Read command-line settings for temporary portfolio Optuna tuning."""

    parser = argparse.ArgumentParser(
        description=(
            "Temporarily tune a global portfolio-wide regression "
            "or XGBoost model with Optuna."
        ),
    )

    parser.add_argument(
        "--model-family",
        required=True,
        choices=[
            "xgboost",
            "regression",
        ],
        help="Model family to tune.",
    )

    parser.add_argument(
        "--base-config",
        default=DEFAULT_BASE_CONFIG,
        help=(
            "Path to shared project configuration YAML. "
            "Default: configs/base.yaml"
        ),
    )

    parser.add_argument(
        "--feature-set",
        default=DEFAULT_FEATURE_SET,
        help=(
            "Feature-set name without .yaml. "
            "Default: weekly_v1"
        ),
    )

    parser.add_argument(
        "--feature-map-path",
        default=DEFAULT_FEATURE_MAP,
        help=(
            "Path to the point-in-time-safe feature map. "
            "Default: data/processed/weekly_features_weekly_v1.csv"
        ),
    )

    parser.add_argument(
        "--version",
        required=True,
        choices=[
            "v1",
            "v2",
        ],
        help=(
            "Validation-window version. "
            "v1 = 2025-01-01 to 2025-12-31; "
            "v2 = 2022-01-01 to 2025-12-31."
        ),
    )

    parser.add_argument(
        "--minimum-training-weeks",
        type=int,
        default=DEFAULT_MINIMUM_TRAINING_WEEKS,
        help=(
            "Minimum historical target weeks before fitting. "
            "Default: 104"
        ),
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help=(
            "Number of new Optuna trials. "
            "Default: 100"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Optuna and model random seed. Default: 42",
    )

    return parser.parse_args()


def get_output_settings(
    model_family: str,
    version: str,
) -> tuple[Path, str]:
    """Return temporary output directory and study name."""

    study_name = (
        f"temp_portfolio_{model_family}_"
        f"mae_optuna_{version}"
    )

    output_directory = Path(
        "outputs/tuning"
    ) / study_name

    return output_directory, study_name


def load_feature_map(
    feature_map_path: str,
    selected_columns: list[str],
) -> pd.DataFrame:
    """Load and validate the existing point-in-time-safe feature map."""

    path = Path(feature_map_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find feature map: {path}"
        )

    feature_map = pd.read_csv(
        path,
        parse_dates=["target_week"],
    )

    required_columns = {
        "target_week",
        "ticker",
        "target",
        *selected_columns,
    }

    missing_columns = (
        required_columns
        - set(feature_map.columns)
    )

    if missing_columns:
        raise ValueError(
            "Feature map is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    feature_map["target_week"] = pd.to_datetime(
        feature_map["target_week"]
    )

    feature_map["ticker"] = feature_map[
        "ticker"
    ].astype(str)

    feature_map = feature_map.sort_values(
        by=[
            "target_week",
            "ticker",
        ]
    ).reset_index(drop=True)

    if feature_map[selected_columns].isna().any().any():
        raise ValueError(
            "Feature map contains missing selected feature values."
        )

    if feature_map["target"].isna().any():
        raise ValueError(
            "Feature map contains missing target values."
        )

    return feature_map


def get_fixed_feature_columns(
    feature_set: str,
) -> list[str]:
    """Return all point-in-time-safe feature columns from the feature set."""

    feature_config = load_feature_set_config(
        feature_set_name=feature_set,
    )

    feature_spec = build_weekly_feature_spec(
        feature_config=feature_config,
    )

    base_config = load_yaml(
        DEFAULT_BASE_CONFIG
    )

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

    portfolio_tickers = list(
        base_config["tickers"].keys()
    )

    benchmark_ticker = base_config["market"][
        "ticker"
    ]

    weekly_frequency = base_config["forecast"][
        "weekly_frequency"
    ]

    portfolio_weekly_close = to_weekly_close(
        daily_close=daily_close[
            portfolio_tickers
        ].copy(),
        weekly_frequency=weekly_frequency,
    )

    benchmark_weekly_close = to_weekly_close(
        daily_close=daily_close[
            benchmark_ticker
        ].dropna()
        .to_frame(),
        weekly_frequency=weekly_frequency,
    )

    portfolio_returns = calculate_returns(
        portfolio_weekly_close
    ).dropna(how="all")

    benchmark_returns = (
        calculate_returns(benchmark_weekly_close)
        .iloc[:, 0]
        .dropna()
    )

    benchmark_returns.name = benchmark_ticker

    raw_feature_map = build_weekly_training_features(
        weekly_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        spec=feature_spec,
    )

    return get_feature_columns(
        raw_feature_map
    )


def build_xgboost_estimator(
    trial: optuna.Trial,
    seed: int,
) -> XGBRegressor:
    """Sample and build one portfolio-wide XGBoost estimator."""

    return XGBRegressor(
        n_estimators=trial.suggest_int(
            "n_estimators",
            40,
            120,
            step=10,
        ),
        max_depth=trial.suggest_int(
            "max_depth",
            2,
            4,
        ),
        learning_rate=trial.suggest_float(
            "learning_rate",
            0.005,
            0.04,
            log=True,
        ),
        subsample=trial.suggest_float(
            "subsample",
            0.6,
            0.9,
        ),
        colsample_bytree=trial.suggest_float(
            "colsample_bytree",
            0.6,
            0.9,
        ),
        reg_alpha=trial.suggest_float(
            "reg_alpha",
            0.001,
            0.2,
            log=True,
        ),
        reg_lambda=trial.suggest_float(
            "reg_lambda",
            0.001,
            0.2,
            log=True,
        ),
        min_child_weight=trial.suggest_int(
            "min_child_weight",
            2,
            8,
        ),
        random_state=seed,
        n_jobs=-1,
        objective="reg:squarederror",
    )


def build_regression_estimator(
    trial: optuna.Trial,
) -> Pipeline:
    """Sample and build one standardized regression pipeline."""

    estimator_name = trial.suggest_categorical(
        "estimator",
        [
            "ridge",
            "lasso",
            "elastic_net",
        ],
    )

    if estimator_name == "ridge":
        estimator = Ridge(
            alpha=trial.suggest_float(
                "ridge_alpha",
                0.01,
                100.0,
                log=True,
            ),
            fit_intercept=True,
        )

    elif estimator_name == "lasso":
        estimator = Lasso(
            alpha=trial.suggest_float(
                "lasso_alpha",
                0.0001,
                0.1,
                log=True,
            ),
            fit_intercept=True,
            max_iter=30000,
            tol=0.0001,
        )

    else:
        estimator = ElasticNet(
            alpha=trial.suggest_float(
                "elastic_net_alpha",
                0.0001,
                0.1,
                log=True,
            ),
            l1_ratio=trial.suggest_float(
                "elastic_net_l1_ratio",
                0.10,
                0.90,
            ),
            fit_intercept=True,
            max_iter=30000,
            tol=0.0001,
        )

    return Pipeline([
        (
            "standard_scaler",
            StandardScaler(),
        ),
        (
            "regressor",
            estimator,
        ),
    ])


def get_estimator(
    model_family: str,
    trial: optuna.Trial,
    seed: int,
):
    """Return one sampled model for the selected family."""

    if model_family == "xgboost":
        return build_xgboost_estimator(
            trial=trial,
            seed=seed,
        )

    if model_family == "regression":
        return build_regression_estimator(
            trial=trial,
        )

    raise ValueError(
        f"Unknown model family: {model_family}"
    )


def evaluate_portfolio_trial(
    trial: optuna.Trial,
    feature_map: pd.DataFrame,
    portfolio_tickers: list[str],
    selected_columns: list[str],
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    minimum_training_weeks: int,
    model_family: str,
    seed: int,
) -> float:
    """Evaluate one expanding-window portfolio-wide Optuna trial."""

    validation_weeks = sorted(
        feature_map.loc[
            (
                feature_map["target_week"]
                >= validation_start
            )
            & (
                feature_map["target_week"]
                <= validation_end
            ),
            "target_week",
        ]
        .drop_duplicates()
        .tolist()
    )

    per_ticker_absolute_errors = {
        ticker: []
        for ticker in portfolio_tickers
    }

    for target_week in validation_weeks:
        train_data = feature_map.loc[
            feature_map["target_week"] < target_week
        ].copy()

        test_data = feature_map.loc[
            feature_map["target_week"] == target_week
        ].copy()

        available_training_weeks = train_data[
            "target_week"
        ].nunique()

        if available_training_weeks < minimum_training_weeks:
            continue

        training_rows = train_data.loc[
            train_data["ticker"].isin(
                portfolio_tickers
            )
        ].copy()

        test_rows = (
            test_data
            .set_index("ticker")
            .reindex(portfolio_tickers)
            .reset_index()
        )

        if test_rows[selected_columns].isna().any().any():
            raise optuna.TrialPruned(
                "One or more portfolio stocks have missing "
                "validation features."
            )

        estimator = get_estimator(
            model_family=model_family,
            trial=trial,
            seed=seed,
        )

        estimator.fit(
            training_rows[selected_columns],
            training_rows["target"],
        )

        predictions = estimator.predict(
            test_rows[selected_columns]
        )

        actual_returns = test_rows["target"].to_numpy()

        absolute_errors = np.abs(
            predictions - actual_returns
        )

        for ticker, absolute_error in zip(
            portfolio_tickers,
            absolute_errors,
        ):
            per_ticker_absolute_errors[ticker].append(
                float(absolute_error)
            )

    ticker_mae = {}

    for ticker, errors in per_ticker_absolute_errors.items():
        if not errors:
            raise optuna.TrialPruned(
                "No valid validation predictions for "
                f"{ticker}."
            )

        ticker_mae[ticker] = float(
            np.mean(errors)
        )

    portfolio_mae = float(
        np.mean(
            list(ticker_mae.values())
        )
    )

    trial.set_user_attr(
        "target_scope",
        "portfolio",
    )

    trial.set_user_attr(
        "training_tickers",
        portfolio_tickers,
    )

    trial.set_user_attr(
        "scored_tickers",
        portfolio_tickers,
    )

    trial.set_user_attr(
        "selected_columns",
        selected_columns,
    )

    trial.set_user_attr(
        "per_ticker_mae",
        ticker_mae,
    )

    return portfolio_mae


def build_best_model_config(
    best_trial: optuna.trial.FrozenTrial,
    model_family: str,
    selected_columns: list[str],
    portfolio_tickers: list[str],
    feature_set: str,
    feature_map_path: str,
    minimum_training_weeks: int,
    seed: int,
    run_id: str,
    version: str,
) -> dict[str, Any]:
    """Convert one temporary best Optuna trial into a normal model YAML."""

    if model_family == "xgboost":
        parameters = {
            "estimator": "xgboost",
            "random_state": seed,
            "n_jobs": -1,
            "n_estimators": best_trial.params[
                "n_estimators"
            ],
            "max_depth": best_trial.params[
                "max_depth"
            ],
            "learning_rate": best_trial.params[
                "learning_rate"
            ],
            "subsample": best_trial.params[
                "subsample"
            ],
            "colsample_bytree": best_trial.params[
                "colsample_bytree"
            ],
            "reg_alpha": best_trial.params[
                "reg_alpha"
            ],
            "reg_lambda": best_trial.params[
                "reg_lambda"
            ],
            "min_child_weight": best_trial.params[
                "min_child_weight"
            ],
            "minimum_training_weeks": (
                minimum_training_weeks
            ),
        }

        model_name = "tree_boosting"
        label = "Temporary Portfolio XGBoost Optuna Best"

    else:
        estimator_name = best_trial.params[
            "estimator"
        ]

        parameters = {
            "standardize": True,
            "fit_intercept": True,
            "max_iter": 30000,
            "tol": 0.0001,
            "estimator": estimator_name,
            "minimum_training_weeks": (
                minimum_training_weeks
            ),
        }

        if estimator_name == "ridge":
            parameters["alpha"] = best_trial.params[
                "ridge_alpha"
            ]

        elif estimator_name == "lasso":
            parameters["alpha"] = best_trial.params[
                "lasso_alpha"
            ]

        else:
            parameters["alpha"] = best_trial.params[
                "elastic_net_alpha"
            ]

            parameters["l1_ratio"] = best_trial.params[
                "elastic_net_l1_ratio"
            ]

        model_name = "general_regression"
        label = "Temporary Portfolio Regression Optuna Best"

    return {
        "model": {
            "name": model_name,
            "label": label,
            "version": f"temp_{version}",
            "run_id": f"{run_id}_best",
        },
        "parameters": parameters,
        "training": {
            "tickers": portfolio_tickers,
        },
        "features": {
            "feature_set": feature_set,
            "feature_map_path": feature_map_path,
            "selected_columns": selected_columns,
        },
    }


def write_yaml(
    path: Path,
    data: dict[str, Any],
) -> None:
    """Write one model configuration YAML."""

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            data,
            file,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )


def main() -> None:
    """Run one temporary portfolio-wide Optuna tuning study."""

    args = parse_arguments()

    if args.n_trials <= 0:
        raise ValueError(
            "--n-trials must be greater than zero."
        )

    if args.minimum_training_weeks <= 0:
        raise ValueError(
            "--minimum-training-weeks must be greater than zero."
        )

    validation_period = VALIDATION_PERIODS[
        args.version
    ]

    validation_start = pd.Timestamp(
        validation_period["start"]
    )

    validation_end = pd.Timestamp(
        validation_period["end"]
    )

    if validation_start > validation_end:
        raise ValueError(
            "--validation-start must not be later than "
            "--validation-end."
        )

    base_config = load_yaml(
        args.base_config
    )

    portfolio_tickers = list(
        base_config["tickers"].keys()
    )

    benchmark_ticker = str(
        base_config["market"]["ticker"]
    )

    if benchmark_ticker in portfolio_tickers:
        raise ValueError(
            "HSI benchmark must not be a portfolio ticker."
        )

    if len(portfolio_tickers) != 10:
        raise ValueError(
            "Expected 10 portfolio stocks, found "
            f"{len(portfolio_tickers)}."
        )

    selected_columns = get_fixed_feature_columns(
        feature_set=args.feature_set,
    )

    feature_map = load_feature_map(
        feature_map_path=args.feature_map_path,
        selected_columns=selected_columns,
    )

    output_directory, study_name = get_output_settings(
        model_family=args.model_family,
        version=args.version,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    storage_path = (
        output_directory
        / "optuna_studies.db"
    )

    storage_url = f"sqlite:///{storage_path}"

    sampler = optuna.samplers.TPESampler(
        seed=args.seed
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        storage=storage_url,
        load_if_exists=True,
    )

    initial_trial_count = len(
        study.trials
    )

    def objective(
        trial: optuna.Trial,
    ) -> float:
        """Evaluate one portfolio-wide model candidate."""

        return evaluate_portfolio_trial(
            trial=trial,
            feature_map=feature_map,
            portfolio_tickers=portfolio_tickers,
            selected_columns=selected_columns,
            validation_start=validation_start,
            validation_end=validation_end,
            minimum_training_weeks=(
                args.minimum_training_weeks
            ),
            model_family=args.model_family,
            seed=args.seed,
        )

    print("\n" + "=" * 70)
    print("TEMPORARY PORTFOLIO OPTUNA TUNING")
    print("=" * 70)
    print(f"Model family: {args.model_family}")
    print(f"Study: {study.study_name}")
    print("Target scope: all 10 portfolio stocks")
    print("Objective: equal-weight portfolio MAE")
    print(f"New trials: {args.n_trials}")
    print(f"Existing trials: {initial_trial_count}")
    print(f"Storage: {storage_url}")
    print(
        "Evaluation period: "
        f"{validation_start.date()} to "
        f"{validation_end.date()}"
    )
    print(
        "Minimum training weeks: "
        f"{args.minimum_training_weeks}"
    )
    print(
        "Fixed feature columns: "
        f"{len(selected_columns)}"
    )
    print("=" * 70)

    start_time = time.perf_counter()

    with tqdm(
        total=args.n_trials,
        desc="Optuna trials",
        unit="trial",
    ) as progress_bar:
        def progress_callback(
            current_study: optuna.Study,
            current_trial: optuna.trial.FrozenTrial,
        ) -> None:
            """Update progress display after each Optuna trial."""

            del current_trial

            progress_bar.n = (
                len(current_study.trials)
                - initial_trial_count
            )

            completed_trials = sum(
                trial.state
                == optuna.trial.TrialState.COMPLETE
                for trial in current_study.trials
            )

            pruned_trials = sum(
                trial.state
                == optuna.trial.TrialState.PRUNED
                for trial in current_study.trials
            )

            progress_bar.set_postfix(
                {
                    "Complete": completed_trials,
                    "Pruned": pruned_trials,
                    "Best": (
                        f"{current_study.best_value:.6f}"
                        if completed_trials
                        else "n/a"
                    ),
                },
                refresh=True,
            )

            progress_bar.refresh()

        study.optimize(
            objective,
            n_trials=args.n_trials,
            callbacks=[progress_callback],
            catch=(ValueError,),
        )

    elapsed_seconds = time.perf_counter() - start_time

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state
        == optuna.trial.TrialState.COMPLETE
    ]

    pruned_trials = [
        trial
        for trial in study.trials
        if trial.state
        == optuna.trial.TrialState.PRUNED
    ]

    failed_trials = [
        trial
        for trial in study.trials
        if trial.state
        == optuna.trial.TrialState.FAIL
    ]

    print("\n" + "=" * 70)
    print("TEMPORARY PORTFOLIO OPTUNA TUNING COMPLETED")
    print("=" * 70)
    print(f"Study name: {study.study_name}")
    print(f"Total trials: {len(study.trials)}")
    print(f"Completed trials: {len(completed_trials)}")
    print(f"Pruned trials: {len(pruned_trials)}")
    print(f"Failed trials: {len(failed_trials)}")
    print(f"Elapsed seconds: {elapsed_seconds:.2f}")

    if not completed_trials:
        raise RuntimeError(
            "No completed Optuna trial was produced."
        )

    best_trial = study.best_trial

    best_model_config = build_best_model_config(
        best_trial=best_trial,
        model_family=args.model_family,
        selected_columns=selected_columns,
        portfolio_tickers=portfolio_tickers,
        feature_set=args.feature_set,
        feature_map_path=args.feature_map_path,
        minimum_training_weeks=(
            args.minimum_training_weeks
        ),
        seed=args.seed,
        run_id=study_name,
        version=args.version,
    )

    trials_csv_path = (
        output_directory
        / "trials.csv"
    )

    study.trials_dataframe().to_csv(
        trials_csv_path,
        index=False,
    )

    best_trial_json_path = (
        output_directory
        / "best_trial.json"
    )

    best_trial_payload = {
        "number": best_trial.number,
        "value": best_trial.value,
        "params": best_trial.params,
        "user_attrs": best_trial.user_attrs,
    }

    with best_trial_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            best_trial_payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    best_model_config_path = (
        output_directory
        / "best_model_config.yaml"
    )

    write_yaml(
        path=best_model_config_path,
        data=best_model_config,
    )

    metadata_path = (
        output_directory
        / "tuning_metadata.json"
    )

    metadata = {
        "study_name": study.study_name,
        "model_family": args.model_family,
        "tuning_version": args.version,
        "target_scope": "portfolio",
        "portfolio_tickers": portfolio_tickers,
        "benchmark_ticker": benchmark_ticker,
        "objective": "equal_weight_portfolio_mae",
        "validation_start": str(
            validation_start.date()
        ),
        "validation_end": str(
            validation_end.date()
        ),
        "minimum_training_weeks": (
            args.minimum_training_weeks
        ),
        "feature_set": args.feature_set,
        "feature_map_path": args.feature_map_path,
        "number_of_fixed_features": len(
            selected_columns
        ),
        "total_trials": len(study.trials),
        "completed_trials": len(completed_trials),
        "pruned_trials": len(pruned_trials),
        "failed_trials": len(failed_trials),
        "best_trial_number": best_trial.number,
        "best_value": best_trial.value,
        "storage_url": storage_url,
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nBest equal-weight portfolio MAE:")
    print(f"{study.best_value:.6f}")

    print("\nBest trial number:")
    print(best_trial.number)

    print("\nBest parameters:")
    for parameter_name, parameter_value in (
        best_trial.params.items()
    ):
        print(f"{parameter_name}: {parameter_value}")

    print("\nBest per-ticker validation MAE:")
    for ticker, ticker_mae in (
        best_trial.user_attrs[
            "per_ticker_mae"
        ].items()
    ):
        print(f"{ticker}: {ticker_mae:.6f}")

    print(f"\nTrials CSV: {trials_csv_path}")
    print(
        "Best trial JSON: "
        f"{best_trial_json_path}"
    )
    print(
        "Best model config YAML: "
        f"{best_model_config_path}"
    )
    print(
        "Tuning metadata JSON: "
        f"{metadata_path}"
    )


if __name__ == "__main__":
    main()