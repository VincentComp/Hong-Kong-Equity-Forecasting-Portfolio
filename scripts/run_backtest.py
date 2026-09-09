"""Run a historical weekly-return backtest for one selected model.

Examples:
    python scripts/run_backtest.py

    python scripts/run_backtest.py \
        --model-config configs/models/moving_average.yaml

    python scripts/run_backtest.py \
        --model-config configs/models/ewma.yaml

    python scripts/run_backtest.py \
        --model-config configs/models/moving_average.yaml \
        --start 2025-01-01 \
        --end 2025-12-31

The default evaluation period is the test period in configs/base.yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from src.hk_equity.data.preprocess import (
    calculate_returns,
    to_weekly_close,
)
from src.hk_equity.evaluation.metrics import (
    calculate_rank_ic_by_week,
    calculate_top_n_return,
    forecast_metrics,
)
from src.hk_equity.models.registry import (
    get_backtest_forecasts,
    get_model_label,
)
from src.hk_equity.utils.config import load_project_config


def parse_arguments() -> argparse.Namespace:
    """Read command-line settings for one backtest run."""

    parser = argparse.ArgumentParser(
        description=(
            "Run historical weekly-return backtest for one selected model."
        )
    )

    parser.add_argument(
        "--base-config",
        default="configs/base.yaml",
        help="Path to shared project configuration YAML.",
    )

    parser.add_argument(
        "--model-config",
        default="configs/models/moving_average.yaml",
        help="Path to one model-specific configuration YAML.",
    )

    parser.add_argument(
        "--start",
        default=None,
        help=(
            "Evaluation start date in YYYY-MM-DD format. "
            "Default: test_start in base.yaml."
        ),
    )

    parser.add_argument(
        "--end",
        default=None,
        help=(
            "Evaluation end date in YYYY-MM-DD format. "
            "Default: test_end in base.yaml."
        ),
    )

    return parser.parse_args()


def load_daily_close_prices(
    raw_data_directory: str,
) -> pd.DataFrame:
    """Load the latest saved 10-stock daily Close-price table."""

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


def create_prediction_table(
    actual_returns: pd.DataFrame,
    predicted_returns: pd.DataFrame,
    tickers: dict[str, str],
    model_label: str,
    model_key: str,
) -> pd.DataFrame:
    """Convert wide actual/predicted return tables into a long audit table."""

    records = []

    for week_ending in actual_returns.index:
        for ticker in actual_returns.columns:
            actual_return = actual_returns.at[
                week_ending,
                ticker,
            ]

            predicted_return = predicted_returns.at[
                week_ending,
                ticker,
            ]

            if pd.isna(actual_return) or pd.isna(predicted_return):
                continue

            error = predicted_return - actual_return

            records.append({
                "week_ending": week_ending.date().isoformat(),
                "ticker": ticker,
                "company": tickers[ticker],
                "model": model_label,
                "model_key": model_key,
                "actual_weekly_return": actual_return,
                "predicted_weekly_return": predicted_return,
                "forecast_error": error,
                "absolute_error": abs(error),
                "squared_error": error ** 2,
                "direction_correct": int(
                    np.sign(predicted_return)
                    == np.sign(actual_return)
                ),
            })

    prediction_table = pd.DataFrame(records)

    if prediction_table.empty:
        raise ValueError(
            "No valid actual/predicted return pairs were created."
        )

    return prediction_table


def main() -> None:
    """Run a time-aligned historical backtest and save all outputs."""

    args = parse_arguments()

    # Load shared project config and selected model config.
    base_config, model_config = load_project_config(
        base_config_path=args.base_config,
        model_config_path=args.model_config,
    )

    model_key = model_config["model"]["name"]
    model_label = get_model_label(model_config)

    # Use command-line dates if supplied; otherwise use official test period.
    evaluation_start = (
        args.start
        if args.start is not None
        else base_config["periods"]["test_start"]
    )

    evaluation_end = (
        args.end
        if args.end is not None
        else base_config["periods"]["test_end"]
    )

    # Load daily prices and convert them to configured weekly Close prices.
    daily_close = load_daily_close_prices(
        raw_data_directory=base_config["paths"]["raw_data"],
    )

    weekly_close = to_weekly_close(
        daily_close=daily_close,
        weekly_frequency=base_config["forecast"]["weekly_frequency"],
    )

    weekly_returns = calculate_returns(
        weekly_close
    )

    # Historical forecast table: model functions are shifted internally,
    # ensuring the forecast at week t does not use the realised return at t.
    all_predicted_returns = get_backtest_forecasts(
        weekly_returns=weekly_returns,
        model_config=model_config,
    )

    actual_returns = weekly_returns.loc[
        evaluation_start:evaluation_end
    ]

    predicted_returns = all_predicted_returns.loc[
        evaluation_start:evaluation_end
    ]

    if actual_returns.empty:
        raise ValueError(
            f"No actual weekly returns in evaluation period: "
            f"{evaluation_start} to {evaluation_end}"
        )

    # Remove dates with no valid prediction for every stock.
    valid_dates = predicted_returns.dropna(
        how="all"
    ).index

    actual_returns = actual_returns.loc[valid_dates]
    predicted_returns = predicted_returns.loc[valid_dates]

    prediction_table = create_prediction_table(
        actual_returns=actual_returns,
        predicted_returns=predicted_returns,
        tickers=base_config["tickers"],
        model_label=model_label,
        model_key=model_key,
    )

    # Calculate model metrics separately for each stock.
    metrics_rows = []

    for ticker in base_config["tickers"]:
        ticker_data = prediction_table[
            prediction_table["ticker"] == ticker
        ]

        ticker_metrics = forecast_metrics(
            actual=ticker_data["actual_weekly_return"],
            predicted=ticker_data["predicted_weekly_return"],
        )

        metrics_rows.append({
            "ticker": ticker,
            "company": base_config["tickers"][ticker],
            "model": model_label,
            "model_key": model_key,
            **ticker_metrics,
        })

    metrics_by_stock = (
        pd.DataFrame(metrics_rows)
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    # Calculate overall metrics using every stock-week forecast observation.
    overall_metrics = forecast_metrics(
        actual=prediction_table["actual_weekly_return"],
        predicted=prediction_table["predicted_weekly_return"],
    )

    overall_metrics_table = pd.DataFrame([{
        "model": model_label,
        "model_key": model_key,
        "evaluation_start": evaluation_start,
        "evaluation_end": evaluation_end,
        **overall_metrics,
    }])

    # Evaluate whether the model ranks relatively stronger stocks correctly.
    rank_ic_by_week = calculate_rank_ic_by_week(
        actual_returns=actual_returns,
        predicted_returns=predicted_returns,
    )

    top_3_by_week = calculate_top_n_return(
        actual_returns=actual_returns,
        predicted_returns=predicted_returns,
        top_n=3,
    )

    # Save model-specific artefacts in a model/date-range folder.
    evaluation_label = (
        f"{evaluation_start}_to_{evaluation_end}"
    )

    output_dir = (
        Path(base_config["paths"]["backtests"])
        / model_key
        / evaluation_label
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_table.to_csv(
        output_dir / "predictions.csv",
        index=False,
    )

    metrics_by_stock.to_csv(
        output_dir / "metrics_by_stock.csv",
        index=False,
    )

    overall_metrics_table.to_csv(
        output_dir / "overall_metrics.csv",
        index=False,
    )

    rank_ic_by_week.to_csv(
        output_dir / "rank_ic_by_week.csv",
        index=False,
    )

    top_3_by_week.to_csv(
        output_dir / "top_3_by_week.csv",
        index=False,
    )

    print("\nBacktest completed successfully.\n")

    print("Overall metrics:")
    print(
        overall_metrics_table.to_string(
            index=False,
        )
    )

    print("\nMetrics by stock:")
    print(
        metrics_by_stock.to_string(
            index=False,
        )
    )

    if not rank_ic_by_week.empty:
        print("\nAverage weekly Rank IC:")
        print(
            f"{rank_ic_by_week['rank_ic'].mean():.4f}"
        )

    if not top_3_by_week.empty:
        print("\nAverage Top-3 minus Equal-Weight return:")
        print(
            f"{top_3_by_week['top_n_minus_equal_weight'].mean():.4%}"
        )

    print(f"\nBacktest output folder: {output_dir}")


if __name__ == "__main__":
    main()