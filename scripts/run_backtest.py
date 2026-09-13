"""Run a historical weekly-return backtest for one selected model.

Examples:
    python scripts/run_backtest.py

    python -m scripts.run_backtest \
        --model-config configs/models/ma_4w_v1.yaml

    python scripts/run_backtest.py \
        --model-config configs/models/ma_4w_v1.yaml \
        --start 2025-01-01 \
        --end 2025-12-31


    #currently use this
    python -m scripts.run_backtest \
        --model-config configs/models/regression_v1.yaml \
        --start 2020-01-01 \
        --end 2026-08-31

The default evaluation period is the test period in configs/base.yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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


#parse the input
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
        default="configs/models/ma_4w_v1.yaml",
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


#Load the latest daily close price as dataframe
def load_daily_close_prices(
    raw_data_directory: str,
) -> pd.DataFrame:
    """Load the latest saved daily Close-price table."""

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


def prepare_portfolio_weekly_returns(
    daily_close: pd.DataFrame,
    portfolio_tickers: list[str],
    weekly_frequency: str,
) -> pd.DataFrame:
    """Create weekly returns using portfolio tickers only."""

    missing_tickers = [
        ticker
        for ticker in portfolio_tickers
        if ticker not in daily_close.columns
    ]

    if missing_tickers:
        raise ValueError(
            "Missing portfolio ticker columns: "
            f"{missing_tickers}"
        )

    portfolio_daily_close = daily_close[
        portfolio_tickers
    ].copy()

    portfolio_weekly_close = to_weekly_close(
        daily_close=portfolio_daily_close,
        weekly_frequency=weekly_frequency,
    )

    portfolio_weekly_returns = calculate_returns(
        portfolio_weekly_close
    ).dropna(how="all")

    if portfolio_weekly_returns.empty:
        raise ValueError(
            "No portfolio weekly returns could be calculated."
        )

    return portfolio_weekly_returns


def prepare_benchmark_weekly_returns(
    daily_close: pd.DataFrame,
    benchmark_ticker: str,
    weekly_frequency: str,
) -> pd.Series:
    """Create weekly benchmark returns, normally using the HSI."""

    if benchmark_ticker not in daily_close.columns:
        raise ValueError(
            "Missing benchmark column in daily data: "
            f"{benchmark_ticker}"
        )

    benchmark_daily_close = daily_close[
        benchmark_ticker
    ].dropna().to_frame()

    benchmark_weekly_close = to_weekly_close(
        daily_close=benchmark_daily_close,
        weekly_frequency=weekly_frequency,
    )

    benchmark_weekly_returns = (
        calculate_returns(benchmark_weekly_close)
        .iloc[:, 0]
        .dropna()
    )

    if benchmark_weekly_returns.empty:
        raise ValueError(
            "No benchmark weekly returns could be calculated."
        )

    benchmark_weekly_returns.name = benchmark_ticker

    return benchmark_weekly_returns


def create_prediction_table(
    actual_returns: pd.DataFrame,
    predicted_returns: pd.DataFrame,
    tickers: dict[str, str],
    model_label: str,
    model_key: str,
    model_run_id: str,
    source_model_profile: dict[str, str] | None = None,
    source_model_run_id: dict[str, str] | None = None,
    source_model_label: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Convert wide actual/predicted return tables into a long audit table."""

    portfolio_tickers = list(tickers.keys())

    unexpected_actual = [
        ticker
        for ticker in actual_returns.columns
        if ticker not in portfolio_tickers
    ]

    unexpected_predicted = [
        ticker
        for ticker in predicted_returns.columns
        if ticker not in portfolio_tickers
    ]

    if unexpected_actual or unexpected_predicted:
        raise ValueError(
            "Benchmark or unexpected ticker entered the prediction table. "
            f"Unexpected actual columns: {unexpected_actual}; "
            f"unexpected predicted columns: {unexpected_predicted}"
        )

    actual_returns = actual_returns.reindex(
        columns=portfolio_tickers
    )

    predicted_returns = predicted_returns.reindex(
        columns=portfolio_tickers,
    )

    records = []

    for week_ending in actual_returns.index:  #for every week
        for ticker in portfolio_tickers: #for every ticker
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

            row = {
                "week_ending": week_ending.date().isoformat(),
                "ticker": ticker,
                "company": tickers[ticker],
                "model": model_label,
                "model_key": model_key,
                "model_run_id": model_run_id,
                "actual_weekly_return": actual_return,
                "predicted_weekly_return": predicted_return,
                "forecast_error": error,
                "absolute_error": abs(error),
                "squared_error": error ** 2,
                "direction_correct": int(
                    np.sign(predicted_return)
                    == np.sign(actual_return)
                ),
            }

            if source_model_profile is not None:
                row["source_model_profile"] = (
                    source_model_profile.get(
                        ticker,
                        model_key,
                    )
                )

            if source_model_run_id is not None:
                row["source_model_run_id"] = (
                    source_model_run_id.get(
                        ticker,
                        model_run_id,
                    )
                )

            if source_model_label is not None:
                row["source_model_label"] = (
                    source_model_label.get(
                        ticker,
                        model_label,
                    )
                )

            records.append(row)

    prediction_table = pd.DataFrame(records)# return the every-week result dataframe

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
    model_run_id = model_config["model"]["run_id"]
    model_label = get_model_label(model_config)

    source_model_profile: dict[str, str] | None = None
    source_model_run_id: dict[str, str] | None = None
    source_model_label: dict[str, str] | None = None

    if model_key == "portfolio_router":
        from src.hk_equity.models.portfolio_router import (
            load_portfolio_router_plan,
        )

        portfolio_tickers = list(
            base_config["tickers"].keys()
        )

        router_plan = load_portfolio_router_plan(
            model_config=model_config,
            portfolio_tickers=portfolio_tickers,
        )

        source_model_profile = {}
        source_model_run_id = {}
        source_model_label = {}

        for ticker, profile_name in (
            router_plan["assignments"].items()
        ):
            profile_details = router_plan["profiles"][
                profile_name
            ]

            child_model_config = profile_details[
                "model_config"
            ]

            child_model_settings = child_model_config[
                "model"
            ]

            source_model_profile[ticker] = profile_name
            source_model_run_id[ticker] = child_model_settings[
                "run_id"
            ]
            source_model_label[ticker] = child_model_settings.get(
                "label",
                child_model_settings["name"],
            )

    # Use command-line dates if supplied; otherwise use official test period.
    evaluation_start = (#get the start date -> if not specify, use base.yaml
        args.start
        if args.start is not None
        else base_config["periods"]["test_start"]
    )

    evaluation_end = (#get the end date -> if not specify, use base.yaml
        args.end
        if args.end is not None
        else base_config["periods"]["test_end"]
    )

    # Load daily prices and convert them to configured weekly Close prices.
    daily_close = load_daily_close_prices(
        raw_data_directory=base_config["paths"]["raw_data"],
    )

    portfolio_tickers = list(
        base_config["tickers"].keys()
    )

    weekly_frequency = base_config["forecast"]["weekly_frequency"]

    # The downloaded CSV may also contain the HSI benchmark. The benchmark is
    # reserved for advanced-model features and is never a portfolio target.
    weekly_returns = prepare_portfolio_weekly_returns(
        daily_close=daily_close,
        portfolio_tickers=portfolio_tickers,
        weekly_frequency=weekly_frequency,
    )

    benchmark_weekly_returns = prepare_benchmark_weekly_returns(
        daily_close=daily_close,
        benchmark_ticker=base_config["market"]["ticker"],
        weekly_frequency=weekly_frequency,
    )

    # Historical forecast table: model functions are shifted internally for
    # baseline models. Regression uses an expanding window and only earlier
    # data, with the benchmark supplied separately from portfolio returns.
    all_predicted_returns = get_backtest_forecasts(
        weekly_returns=weekly_returns,
        benchmark_returns=benchmark_weekly_returns,
        model_config=model_config,
    )

    #only get use the data within teesting period for test
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

    #create the prediction model table
    prediction_table = create_prediction_table(
        actual_returns=actual_returns,
        predicted_returns=predicted_returns,
        tickers=base_config["tickers"],
        model_label=model_label,
        model_key=model_key,
        model_run_id=model_run_id,
        source_model_profile=source_model_profile,
        source_model_run_id=source_model_run_id,
        source_model_label=source_model_label,
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

        row = {
            "ticker": ticker,
            "company": base_config["tickers"][ticker],
            "model": model_label,
            "model_key": model_key,
            "model_run_id": model_run_id,
        }

        if source_model_profile is not None:
            row["source_model_profile"] = (
                source_model_profile.get(
                    ticker,
                    model_key,
                )
            )

        if source_model_run_id is not None:
            row["source_model_run_id"] = (
                source_model_run_id.get(
                    ticker,
                    model_run_id,
                )
            )

        if source_model_label is not None:
            row["source_model_label"] = (
                source_model_label.get(
                    ticker,
                    model_label,
                )
            )

        row.update(ticker_metrics)

        metrics_rows.append(row)

    metrics_by_stock = (#sort the result with MAE
        pd.DataFrame(metrics_rows)
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    #Build the Evaluation metric for the entire portfolio
    overall_metrics = forecast_metrics(
        actual=prediction_table["actual_weekly_return"],
        predicted=prediction_table["predicted_weekly_return"],
    )

    overall_metrics_table = pd.DataFrame([{
        "model": model_label,
        "model_key": model_key,
        "model_run_id": model_run_id,
        "model_version": model_config["model"]["version"],
        "model_parameters": str(
            model_config.get("parameters", {})
        ),
        "model_features": str(
            model_config.get("features", {})
        ),
        "evaluation_start": evaluation_start,
        "evaluation_end": evaluation_end,
        **overall_metrics,
    }])

    # Evaluate whether the model ranks relatively stronger stocks correctly.
    # Rank IC and Top-3 evaluation require one valid forecast for every
    # portfolio stock in the same week.
    complete_portfolio_dates = predicted_returns.dropna(
        how="any"
    ).index

    portfolio_actual_returns = actual_returns.loc[
        complete_portfolio_dates
    ]

    portfolio_predicted_returns = predicted_returns.loc[
        complete_portfolio_dates
    ]

    rank_ic_by_week = calculate_rank_ic_by_week(
        actual_returns=portfolio_actual_returns,
        predicted_returns=portfolio_predicted_returns,
    )

    top_3_by_week = calculate_top_n_return(
        actual_returns=portfolio_actual_returns,
        predicted_returns=portfolio_predicted_returns,
        top_n=3,
    )

    # Save model-specific artefacts in a model/date-range folder.
    evaluation_label = (
        f"{evaluation_start}_to_{evaluation_end}"
    )

    output_dir = (
        Path(base_config["paths"]["backtests"])
        / model_run_id
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