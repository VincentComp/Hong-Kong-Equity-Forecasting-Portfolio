from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.hk_equity.data.preprocess import (
    calculate_returns,
    to_weekly_close,
)
from src.hk_equity.features.weekly_features import (
    WeeklyFeatureSpec,
    build_live_features,
    build_weekly_training_features,
    drop_incomplete_feature_rows,
    get_feature_columns,
)


CONFIG_PATH = Path("configs/base.yaml")


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    daily_close_path = (
        Path(config["paths"]["raw_data"])
        / "latest_daily_close.csv"
    )

    daily_close = pd.read_csv(
        daily_close_path,
        index_col="Date",
        parse_dates=True,
    ).sort_index()

    portfolio_tickers = list(
        config["tickers"].keys()
    )

    benchmark_ticker = config["market"]["ticker"]
    weekly_frequency = config["forecast"]["weekly_frequency"]

    missing_portfolio = [
        ticker
        for ticker in portfolio_tickers
        if ticker not in daily_close.columns
    ]

    if missing_portfolio:
        raise ValueError(
            f"Missing portfolio columns: {missing_portfolio}"
        )

    if benchmark_ticker not in daily_close.columns:
        raise ValueError(
            f"Missing benchmark column: {benchmark_ticker}"
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

    portfolio_weekly_returns = (
        calculate_returns(portfolio_weekly_close)
        .dropna(how="all")
    )

    benchmark_weekly_returns = (
        calculate_returns(benchmark_weekly_close)
        .iloc[:, 0]
        .dropna()
    )

    spec = WeeklyFeatureSpec(
        include_ratio_features=False,
        include_cross_sectional_features=True,
        winsorize_features=False,
    )

    training_features = build_weekly_training_features(
        weekly_returns=portfolio_weekly_returns,
        benchmark_returns=benchmark_weekly_returns,
        spec=spec,
    )

    feature_columns = get_feature_columns(
        training_features
    )

    clean_training_features = drop_incomplete_feature_rows(
        feature_table=training_features,
        feature_columns=feature_columns,
    )

    live_features = build_live_features(
        weekly_returns=portfolio_weekly_returns,
        benchmark_returns=benchmark_weekly_returns,
        spec=spec,
    )

    required_columns = [
        "target_week",
        "ticker",
        "target",
        "market_return_lag_1",
        "return_lag_52",
        "rolling_mean_26",
        "excess_return_lag_1",
        "momentum_spread_4_12",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in training_features.columns
    ]

    if missing_columns:
        raise AssertionError(
            f"Missing expected feature columns: {missing_columns}"
        )

    if training_features["ticker"].nunique() != len(
        portfolio_tickers
    ):
        raise AssertionError(
            "Training features do not contain all portfolio tickers."
        )

    if live_features["ticker"].nunique() != len(
        portfolio_tickers
    ):
        raise AssertionError(
            "Live features do not contain all portfolio tickers."
        )

    if clean_training_features.empty:
        raise AssertionError(
            "No complete training rows remain after dropping NaNs."
        )

    print("Feature pipeline passed.")
    print(f"Raw training shape: {training_features.shape}")
    print(
        "Clean training shape: "
        f"{clean_training_features.shape}"
    )
    print(f"Live feature shape: {live_features.shape}")
    print(f"Feature count: {len(feature_columns)}")
    print(f"Portfolio tickers: {portfolio_tickers}")
    print(f"Benchmark: {benchmark_ticker}")
    print("\nFeature columns:")
    print(feature_columns)


if __name__ == "__main__":
    main()