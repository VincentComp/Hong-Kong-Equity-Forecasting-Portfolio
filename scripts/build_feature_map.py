"""Build and save a point-in-time-safe weekly regression feature map.

Examples:
    python -m scripts.build_feature_map

    python -m scripts.build_feature_map \
        --feature-set weekly_v1

    python -m scripts.build_feature_map \
        --base-config configs/base.yaml \
        --feature-set weekly_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.hk_equity.data.preprocess import (
    calculate_returns,
    to_weekly_close,
)
from src.hk_equity.features.feature_config import (
    build_weekly_feature_spec,
    get_feature_set_version,
    load_feature_set_config,
)
from src.hk_equity.features.weekly_features import (
    build_weekly_training_features,
    drop_incomplete_feature_rows,
    get_feature_columns,
)
from src.hk_equity.utils.config import load_yaml


#parse the input
def parse_arguments() -> argparse.Namespace:
    """Read command-line settings for one feature-map build."""

    parser = argparse.ArgumentParser(
        description=(
            "Build and save a point-in-time-safe weekly feature map."
        )
    )

    parser.add_argument(
        "--base-config",
        default="configs/base.yaml",
        help="Path to shared project configuration YAML.",
    )

    parser.add_argument(
        "--feature-set",
        default="weekly_v1",
        help=(
            "Feature-set YAML name without .yaml. "
            "Default: weekly_v1."
        ),
    )

    return parser.parse_args()


#Load the latest daily close price as dataframe
def load_daily_close_prices(
    raw_data_directory: str,
) -> pd.DataFrame:
    """Load latest daily Close prices containing stocks and benchmark."""

    daily_close_path = (
        Path(raw_data_directory)
        / "latest_daily_close.csv"
    )

    if not daily_close_path.exists():
        raise FileNotFoundError(
            f"Cannot find: {daily_close_path}\n"
            "Run scripts/run_data_pipeline.py first."
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

    return daily_close


def prepare_weekly_returns(
    daily_close: pd.DataFrame,
    portfolio_tickers: list[str],
    benchmark_ticker: str,
    weekly_frequency: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare portfolio and benchmark weekly returns separately."""

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
            "Missing benchmark ticker column: "
            f"{benchmark_ticker}"
        )

    portfolio_daily_close = daily_close[
        portfolio_tickers
    ].copy()

    benchmark_daily_close = daily_close[
        benchmark_ticker
    ].dropna().to_frame()

    portfolio_weekly_close = to_weekly_close(
        daily_close=portfolio_daily_close,
        weekly_frequency=weekly_frequency,
    )

    benchmark_weekly_close = to_weekly_close(
        daily_close=benchmark_daily_close,
        weekly_frequency=weekly_frequency,
    )

    portfolio_weekly_returns = calculate_returns(
        portfolio_weekly_close
    ).dropna(how="all")

    benchmark_weekly_returns = (
        calculate_returns(benchmark_weekly_close)
        .iloc[:, 0]
        .dropna()
    )

    benchmark_weekly_returns.name = benchmark_ticker

    if portfolio_weekly_returns.empty:
        raise ValueError(
            "No portfolio weekly returns could be calculated."
        )

    if benchmark_weekly_returns.empty:
        raise ValueError(
            "No benchmark weekly returns could be calculated."
        )

    return (
        portfolio_weekly_returns,
        benchmark_weekly_returns,
    )


def save_feature_map(
    feature_map: pd.DataFrame,
    metadata: dict,
    processed_data_directory: str,
    feature_set_name: str,
) -> tuple[Path, Path, Path]:
    """Save processed feature map as CSV, and metadata JSON."""

    output_directory = Path(
        processed_data_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = output_directory / (
        f"weekly_features_{feature_set_name}.csv"
    )


    metadata_path = output_directory / (
        f"weekly_features_{feature_set_name}_metadata.json"
    )

    feature_map.to_csv(
        csv_path,
        index=False,
    )

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

    return csv_path, metadata_path


def main() -> None:
    """Build one complete clean feature map for supervised learning."""

    args = parse_arguments()

    base_config = load_yaml(
        args.base_config
    )

    feature_config = load_feature_set_config(
        feature_set_name=args.feature_set,
    )

    feature_spec = build_weekly_feature_spec(
        feature_config=feature_config,
    )

    feature_set_version = get_feature_set_version(
        feature_config=feature_config,
    )

    portfolio_tickers = list(
        base_config["tickers"].keys()
    )

    benchmark_ticker = base_config["market"]["ticker"]

    weekly_frequency = base_config["forecast"][
        "weekly_frequency"
    ]

    daily_close = load_daily_close_prices(
        raw_data_directory=base_config["paths"]["raw_data"],
    )

    portfolio_weekly_returns, benchmark_weekly_returns = (
        prepare_weekly_returns(
            daily_close=daily_close,
            portfolio_tickers=portfolio_tickers,
            benchmark_ticker=benchmark_ticker,
            weekly_frequency=weekly_frequency,
        )
    )

    # Build all historical rows. Every row's feature values are created from
    # data available before its target_week because weekly_features.py uses
    # lagged returns and lagged rolling calculations.
    raw_feature_map = build_weekly_training_features(
        weekly_returns=portfolio_weekly_returns,
        benchmark_returns=benchmark_weekly_returns,
        spec=feature_spec,
    )

    feature_columns = get_feature_columns(
        raw_feature_map
    )

    # Remove warm-up rows where long lags or rolling windows do not yet have
    # sufficient prior data. The saved map is immediately usable for model.fit.
    feature_map = drop_incomplete_feature_rows(
        feature_table=raw_feature_map,
        feature_columns=feature_columns,
    )

    if feature_map.empty:
        raise ValueError(
            "No complete rows remain in the processed feature map."
        )

    feature_map["target_week"] = pd.to_datetime(
        feature_map["target_week"]
    )

    feature_map = feature_map.sort_values(
        by=[
            "target_week",
            "ticker",
        ]
    ).reset_index(drop=True)

    metadata = {
        "feature_set_name": args.feature_set,
        "feature_set_version": feature_set_version,
        "feature_set_description": feature_config[
            "feature_set"
        ].get(
            "description",
            "",
        ),
        "weekly_frequency": weekly_frequency,
        "benchmark_ticker": benchmark_ticker,
        "portfolio_tickers": portfolio_tickers,
        "number_of_portfolio_tickers": len(
            portfolio_tickers
        ),
        "raw_feature_rows": len(raw_feature_map),
        "complete_feature_rows": len(feature_map),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "target_column": "target",
        "target_week_column": "target_week",
        "ticker_column": "ticker",
        "first_target_week": (
            feature_map["target_week"]
            .min()
            .date()
            .isoformat()
        ),
        "last_target_week": (
            feature_map["target_week"]
            .max()
            .date()
            .isoformat()
        ),
        "generation": feature_config["generation"],
    }

    csv_path, metadata_path = save_feature_map(
        feature_map=feature_map,
        metadata=metadata,
        processed_data_directory=base_config["paths"][
            "processed_data"
        ],
        feature_set_name=args.feature_set,
    )

    print("\nFeature map built successfully.\n")
    print(f"Feature set: {args.feature_set}")
    print(f"Feature version: {feature_set_version}")
    print(f"Benchmark: {benchmark_ticker}")
    print(f"Portfolio stocks: {len(portfolio_tickers)}")
    print(f"Feature columns: {len(feature_columns)}")
    print(f"Raw feature rows: {len(raw_feature_map)}")
    print(f"Complete feature rows: {len(feature_map)}")
    print(
        "Target-week range: "
        f"{metadata['first_target_week']} to "
        f"{metadata['last_target_week']}"
    )
    print(f"\nCSV feature map: {csv_path}")
    print(f"Metadata JSON: {metadata_path}")


if __name__ == "__main__":
    main()