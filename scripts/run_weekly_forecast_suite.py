"""Run selected weekly forecasting models and build one prediction comparison.

Examples:
    python -m scripts.run_weekly_forecast_suite \
        --run-ids \
        xgboost_optuna_v1_stable_portfolio \
        xgboost_portfolio_mae_optuna_v1_best \
        regression_portfolio_mae_optuna_v1_best \
        global_ensemble_v1 \
        zero_v1

    python -m scripts.run_weekly_forecast_suite \
        --as-of 2026-09-18 \
        --run-ids \
        xgboost_optuna_v1_stable_portfolio \
        xgboost_portfolio_mae_optuna_v1_best \
        regression_portfolio_mae_optuna_v1_best \
        global_ensemble_v1 \
        zero_v1

    python -m scripts.run_weekly_forecast_suite \
        --as-of 2026-09-18 \
        --run-ids \
        global_ensemble_v1 \
        --overwrite

This script:
1. Recursively discovers model YAML files in configs/models/**/*.yaml.
2. Filters models by model.run_id.
3. Runs one weekly forecast for each selected model.
4. Does not overwrite existing model forecasts unless --overwrite is used.
5. Builds one ticker-by-model prediction comparison CSV.
6. Keeps ^HSI out of all portfolio prediction tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd

from src.hk_equity.utils.config import load_yaml


def parse_arguments() -> argparse.Namespace:
    """Read command-line settings for the weekly forecast suite."""

    parser = argparse.ArgumentParser(
        description=(
            "Run selected weekly forecasting models and create "
            "one prediction comparison table."
        ),
    )

    parser.add_argument(
        "--as-of",
        default=None,
        help=(
            "Latest date whose data may be used, in YYYY-MM-DD format. "
            "Default: latest date in latest_daily_close.csv."
        ),
    )

    parser.add_argument(
        "--base-config",
        default="configs/base.yaml",
        help="Path to shared project configuration YAML.",
    )

    parser.add_argument(
        "--models-directory",
        default="configs/models",
        help=(
            "Root directory searched recursively for model YAML configs. "
            "Default: configs/models"
        ),
    )

    parser.add_argument(
        "--run-ids",
        nargs="+",
        required=True,
        help=(
            "Model run_ids to forecast. Each must match "
            "model.run_id in a model YAML config."
        ),
    )

    parser.add_argument(
        "--output-name",
        default="weekly_forecast_comparison",
        help=(
            "Comparison CSV name without extension. "
            "Default: weekly_forecast_comparison"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow replacement of existing per-model forecasts and "
            "the comparison CSV for the same forecast date."
        ),
    )

    return parser.parse_args()


def discover_model_configs(
    models_directory: str,
    selected_run_ids: list[str],
) -> dict[str, str]:
    """Find exactly one model YAML config for every requested run ID."""

    models_path = Path(
        models_directory
    )

    if not models_path.exists():
        raise FileNotFoundError(
            f"Cannot find models directory: {models_path}"
        )

    yaml_paths = sorted(
        list(models_path.rglob("*.yaml"))
        + list(models_path.rglob("*.yml"))
    )

    if not yaml_paths:
        raise FileNotFoundError(
            f"No model YAML files found in: {models_path}"
        )

    requested_run_ids = list(
        dict.fromkeys(selected_run_ids)
    )

    discovered_configs: dict[str, str] = {}
    duplicate_run_ids: dict[str, list[str]] = {}

    for yaml_path in yaml_paths:
        model_config = load_yaml(
            str(yaml_path)
        )

        if "model" not in model_config:
            continue

        model_settings = model_config["model"]

        if not isinstance(model_settings, dict):
            continue

        run_id = str(
            model_settings.get(
                "run_id",
                "",
            )
        )

        if run_id not in requested_run_ids:
            continue

        if run_id in discovered_configs:
            duplicate_run_ids.setdefault(
                run_id,
                [
                    discovered_configs[run_id],
                ],
            ).append(
                str(yaml_path)
            )

        else:
            discovered_configs[run_id] = str(
                yaml_path
            )

    if duplicate_run_ids:
        duplicate_message = "\n".join(
            f"{run_id}: {paths}"
            for run_id, paths in duplicate_run_ids.items()
        )

        raise ValueError(
            "More than one model YAML has the same requested "
            f"run_id:\n{duplicate_message}"
        )

    missing_run_ids = [
        run_id
        for run_id in requested_run_ids
        if run_id not in discovered_configs
    ]

    if missing_run_ids:
        raise ValueError(
            "Cannot find model YAML config for run_ids: "
            f"{missing_run_ids}"
        )

    return {
        run_id: discovered_configs[run_id]
        for run_id in requested_run_ids
    }


def build_forecast_command(
    base_config_path: str,
    model_config_path: str,
    as_of: str | None,
    overwrite: bool,
) -> list[str]:
    """Build the command used to run one model's weekly forecast."""

    command = [
        sys.executable,
        "-m",
        "scripts.run_weekly_forecast",
        "--base-config",
        base_config_path,
        "--model-config",
        model_config_path,
    ]

    if as_of is not None:
        command.extend([
            "--as-of",
            as_of,
        ])

    if overwrite:
        command.append(
            "--overwrite"
        )

    return command


def run_weekly_forecast(
    base_config_path: str,
    model_config_path: str,
    as_of: str | None,
    overwrite: bool,
) -> None:
    """Run one weekly forecast through the existing forecast script."""

    command = build_forecast_command(
        base_config_path=base_config_path,
        model_config_path=model_config_path,
        as_of=as_of,
        overwrite=overwrite,
    )

    print("\nRunning weekly forecast:")
    print(" ".join(command))

    subprocess.run(
        command,
        check=True,
    )


def get_forecast_date(
    base_config: dict,
    as_of: str | None,
) -> str:
    """Return the expected forecast-date folder label."""

    if as_of is not None:
        return pd.Timestamp(as_of).date().isoformat()

    daily_close_path = (
        Path(base_config["paths"]["raw_data"])
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

    return daily_close.index.max().date().isoformat()


def get_forecast_csv_path(
    forecasts_directory: str,
    forecast_date: str,
    model_run_id: str,
) -> Path:
    """Return the expected saved forecast CSV path."""

    return (
        Path(forecasts_directory)
        / forecast_date
        / model_run_id
        / "forecast.csv"
    )


def load_model_forecast(
    forecast_path: Path,
    model_run_id: str,
    portfolio_tickers: list[str],
) -> pd.DataFrame:
    """Load and validate one saved per-model forecast table."""

    if not forecast_path.exists():
        raise FileNotFoundError(
            "Expected forecast CSV was not created: "
            f"{forecast_path}"
        )

    forecast_table = pd.read_csv(
        forecast_path
    )

    required_columns = {
        "ticker",
        "company",
        "predicted_weekly_return",
    }

    missing_columns = (
        required_columns
        - set(forecast_table.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Forecast output for {model_run_id} is missing "
            f"columns: {sorted(missing_columns)}"
        )

    forecast_table["ticker"] = forecast_table[
        "ticker"
    ].astype(str)

    unexpected_tickers = [
        ticker
        for ticker in forecast_table["ticker"]
        if ticker not in portfolio_tickers
    ]

    if unexpected_tickers:
        raise ValueError(
            f"Forecast output for {model_run_id} contains "
            "benchmark or unexpected tickers: "
            f"{unexpected_tickers}"
        )

    missing_tickers = [
        ticker
        for ticker in portfolio_tickers
        if ticker not in set(
            forecast_table["ticker"]
        )
    ]

    if missing_tickers:
        raise ValueError(
            f"Forecast output for {model_run_id} is missing "
            f"portfolio tickers: {missing_tickers}"
        )

    if forecast_table["ticker"].duplicated().any():
        duplicate_tickers = forecast_table.loc[
            forecast_table["ticker"].duplicated(),
            "ticker",
        ].tolist()

        raise ValueError(
            f"Forecast output for {model_run_id} contains "
            f"duplicate tickers: {duplicate_tickers}"
        )

    selected_columns = [
        "ticker",
        "company",
        "predicted_weekly_return",
    ]

    forecast_table = forecast_table[
        selected_columns
    ].copy()

    forecast_table = forecast_table.rename(
        columns={
            "predicted_weekly_return": model_run_id,
        }
    )

    return forecast_table


def build_prediction_comparison(
    forecast_tables: list[pd.DataFrame],
    portfolio_tickers: list[str],
) -> pd.DataFrame:
    """Combine independent model forecasts into one ticker-by-model table."""

    if not forecast_tables:
        raise ValueError(
            "Cannot build a comparison without forecast tables."
        )

    comparison = pd.DataFrame({
        "ticker": portfolio_tickers,
    })

    company_table = forecast_tables[0][
        [
            "ticker",
            "company",
        ]
    ].copy()

    comparison = comparison.merge(
        company_table,
        on="ticker",
        how="left",
        validate="one_to_one",
    )

    for forecast_table in forecast_tables:
        prediction_columns = [
            column
            for column in forecast_table.columns
            if column not in {
                "ticker",
                "company",
            }
        ]

        if len(prediction_columns) != 1:
            raise ValueError(
                "Each forecast table must contain exactly one "
                "prediction column."
            )

        comparison = comparison.merge(
            forecast_table[
                [
                    "ticker",
                    prediction_columns[0],
                ]
            ],
            on="ticker",
            how="left",
            validate="one_to_one",
        )

    prediction_columns = [
        column
        for column in comparison.columns
        if column not in {
            "ticker",
            "company",
        }
    ]

    if comparison[prediction_columns].isna().any().any():
        raise ValueError(
            "Prediction comparison contains missing model forecasts."
        )

    return comparison


def save_prediction_comparison(
    comparison: pd.DataFrame,
    forecasts_directory: str,
    forecast_date: str,
    output_name: str,
    overwrite: bool,
) -> Path:
    """Save one combined ticker-by-model prediction comparison CSV."""

    comparison_directory = (
        Path(forecasts_directory)
        / forecast_date
        / "comparison"
    )

    comparison_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_path = (
        comparison_directory
        / f"{output_name}.csv"
    )

    if comparison_path.exists() and not overwrite:
        raise FileExistsError(
            "Comparison output already exists: "
            f"{comparison_path}\n"
            "Use --overwrite only for draft/test runs."
        )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    return comparison_path


def main() -> None:
    """Run selected forecasts and save one comparison table."""

    args = parse_arguments()

    base_config = load_yaml(
        args.base_config
    )

    portfolio_tickers = list(
        base_config["tickers"].keys()
    )

    market_ticker = str(
        base_config["market"]["ticker"]
    )

    if market_ticker in portfolio_tickers:
        raise ValueError(
            "Benchmark ticker must not be in portfolio tickers: "
            f"{market_ticker}"
        )

    model_configs = discover_model_configs(
        models_directory=args.models_directory,
        selected_run_ids=args.run_ids,
    )

    forecast_date = get_forecast_date(
        base_config=base_config,
        as_of=args.as_of,
    )

    forecasts_directory = base_config["paths"][
        "forecasts"
    ]

    print("\n" + "=" * 70)
    print("WEEKLY FORECAST SUITE")
    print("=" * 70)
    print(f"Forecast data cutoff: {forecast_date}")
    print(f"Models selected: {len(model_configs)}")
    print(f"Portfolio stocks: {len(portfolio_tickers)}")
    print(f"Benchmark excluded: {market_ticker}")
    print("=" * 70)

    forecast_tables = []

    for position, (
        model_run_id,
        model_config_path,
    ) in enumerate(
        model_configs.items(),
        start=1,
    ):
        print("\n" + "=" * 70)
        print(
            f"FORECAST MODEL {position}/{len(model_configs)}"
        )
        print("=" * 70)
        print(f"Run ID: {model_run_id}")
        print(f"Model config: {model_config_path}")

        expected_forecast_path = get_forecast_csv_path(
            forecasts_directory=forecasts_directory,
            forecast_date=forecast_date,
            model_run_id=model_run_id,
        )

        if expected_forecast_path.exists() and not args.overwrite:
            print(
                "\nSkipping existing forecast: "
                f"{expected_forecast_path}"
            )

        else:
            run_weekly_forecast(
                base_config_path=args.base_config,
                model_config_path=model_config_path,
                as_of=args.as_of,
                overwrite=args.overwrite,
            )

        forecast_table = load_model_forecast(
            forecast_path=expected_forecast_path,
            model_run_id=model_run_id,
            portfolio_tickers=portfolio_tickers,
        )

        forecast_tables.append(
            forecast_table
        )

    comparison = build_prediction_comparison(
        forecast_tables=forecast_tables,
        portfolio_tickers=portfolio_tickers,
    )

    comparison_path = save_prediction_comparison(
        comparison=comparison,
        forecasts_directory=forecasts_directory,
        forecast_date=forecast_date,
        output_name=args.output_name,
        overwrite=args.overwrite,
    )

    print("\n" + "=" * 70)
    print("WEEKLY FORECAST SUITE COMPLETED")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False,
        )
    )

    print(
        "\nPrediction comparison CSV: "
        f"{comparison_path}"
    )


if __name__ == "__main__":
    main()