"""Generate and save a general Project 1 weekly-return forecast.

Examples:
    python scripts/run_weekly_forecast.py

    python scripts/run_weekly_forecast.py \
        --as-of 2026-09-18

    python scripts/run_weekly_forecast.py \
        --as-of 2026-09-18 \
        --model-config configs/models/ewma.yaml

    python scripts/run_weekly_forecast.py \
        --as-of 2026-09-18 \
        --model-config configs/models/ewma.yaml \
        --overwrite

Run this after the final trading day of a completed week. The script detects
the last completed Friday-ending week and forecasts the following week.

No individual course submission dates are hard-coded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.hk_equity.data.context import build_forecast_context
from src.hk_equity.data.preprocess import (
    calculate_returns,
    to_weekly_close,
)
from src.hk_equity.evaluation.forecast_table import build_forecast_table
from src.hk_equity.models.registry import (
    get_model_forecast,
    get_model_label,
)
from src.hk_equity.utils.config import load_project_config
from src.hk_equity.utils.io import (
    export_submission_file,
    save_forecast_run,
)


def parse_arguments() -> argparse.Namespace:
    """Read command-line settings for one forecast run."""

    parser = argparse.ArgumentParser(
        description=(
            "Create next-week return forecasts from the latest saved "
            "daily Close prices."
        )
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
        "--model-config",
        default="configs/models/ma_4w.yaml",
        help="Path to one model-specific configuration YAML.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow replacement of an existing forecast output for the "
            "same forecast date and model. Use only for draft/test runs."
        ),
    )

    return parser.parse_args()


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

    if not isinstance(daily_close.index, pd.DatetimeIndex):
        raise TypeError(
            "The daily close index must be a DatetimeIndex."
        )

    return daily_close


def prepare_portfolio_daily_close(
    daily_close: pd.DataFrame,
    portfolio_tickers: list[str],
    market_ticker: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return portfolio prices and the market benchmark prices.

    The market benchmark is used only to create benchmark returns for models
    such as general_regression. It is not forecasted or submitted as a stock.
    """

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

    if market_ticker not in daily_close.columns:
        raise ValueError(
            f"Missing market ticker column: {market_ticker}"
        )

    portfolio_daily_close = daily_close[
        portfolio_tickers
    ].copy()

    market_daily_close = daily_close[
        market_ticker
    ].copy()

    return portfolio_daily_close, market_daily_close


def build_benchmark_weekly_returns(
    market_daily_close: pd.Series,
    data_cutoff: pd.Timestamp,
    weekly_frequency: str,
) -> pd.Series:
    """Build benchmark weekly returns using data up to the forecast cutoff."""

    market_daily_close = market_daily_close.loc[
        :data_cutoff
    ]

    if market_daily_close.empty:
        raise ValueError(
            "No market benchmark data is available "
            "before data_cutoff."
        )

    market_weekly_close = to_weekly_close(
        daily_close=market_daily_close.to_frame(),
        weekly_frequency=weekly_frequency,
    )

    benchmark_weekly_returns = (
        calculate_returns(market_weekly_close)
        .iloc[:, 0]
        .dropna()
    )

    benchmark_weekly_returns.name = "benchmark_return"

    if benchmark_weekly_returns.empty:
        raise ValueError(
            "Could not construct benchmark weekly returns."
        )

    if not isinstance(
        benchmark_weekly_returns.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "benchmark_weekly_returns must have "
            "a DatetimeIndex."
        )

    return benchmark_weekly_returns


def main() -> None:
    """Create, save, and display one model's next-week forecast."""

    args = parse_arguments()

    base_config, model_config = load_project_config(
        base_config_path=args.base_config,
        model_config_path=args.model_config,
    )

    daily_close = load_daily_close_prices(
        raw_data_directory=base_config["paths"]["raw_data"],
    )

    portfolio_tickers = list(
        base_config["tickers"].keys()
    )

    market_ticker = base_config["market"]["ticker"]

    (
        portfolio_daily_close,
        market_daily_close,
    ) = prepare_portfolio_daily_close(
        daily_close=daily_close,
        portfolio_tickers=portfolio_tickers,
        market_ticker=market_ticker,
    )

    data_cutoff = (
        pd.Timestamp(args.as_of)
        if args.as_of is not None
        else portfolio_daily_close.index.max()
    )

    if data_cutoff > daily_close.index.max():
        raise ValueError(
            "--as-of is later than the latest available data date: "
            f"{daily_close.index.max().date()}"
        )

    context = build_forecast_context(
        daily_close=portfolio_daily_close,
        data_cutoff=data_cutoff,
        weekly_frequency=base_config["forecast"][
            "weekly_frequency"
        ],
        horizon_trading_days=base_config["forecast"][
            "horizon_trading_days"
        ],
    )

    benchmark_weekly_returns = build_benchmark_weekly_returns(
        market_daily_close=market_daily_close,
        data_cutoff=data_cutoff,
        weekly_frequency=base_config["forecast"][
            "weekly_frequency"
        ],
    )

    predicted_return = get_model_forecast(
        weekly_returns=context.weekly_returns,
        model_config=model_config,
        benchmark_returns=benchmark_weekly_returns,
    )

    model_key = model_config["model"]["name"]
    model_run_id = model_config["model"]["run_id"]

    model_label = get_model_label(
        model_config=model_config,
    )

    forecast_table = build_forecast_table(
        predicted_return=predicted_return,
        tickers=base_config["tickers"],
        context=context,
        model_config=model_config,
        model_label=model_label,
    )

    forecast_date = context.data_cutoff.date().isoformat()

    metadata = {
        "project_name": base_config["project_name"],
        "data_source": base_config["data"]["source"],
        "price_field": base_config["data"]["price_field"],
        "auto_adjust": base_config["data"]["auto_adjust"],
        "market_ticker": market_ticker,
        "weekly_frequency": base_config["forecast"][
            "weekly_frequency"
        ],
        "horizon_trading_days": base_config["forecast"][
            "horizon_trading_days"
        ],
        "model_config_path": args.model_config,
        "model_key": model_key,
        "model_run_id": model_run_id,
        "model_version": model_config["model"]["version"],
        "model_parameters": model_config.get(
            "parameters",
            {},
        ),
        "requested_data_cutoff": (
            context.data_cutoff.date().isoformat()
        ),
        "actual_last_available_data_date": (
            context.latest_data_date.date().isoformat()
        ),
        "last_completed_week_end": (
            context.latest_completed_week.date().isoformat()
        ),
        "target_week_start": (
            context.target_week_start.date().isoformat()
        ),
        "target_week_end": (
            context.target_week_end.date().isoformat()
        ),
        "forecast_target": (
            "Next Friday-ending weekly stock return"
        ),
        "number_of_stocks": len(forecast_table),
    }

    run_directory = save_forecast_run(
        forecast_df=forecast_table,
        metadata=metadata,
        forecast_date=forecast_date,
        model_key=model_run_id,
        forecasts_dir=base_config["paths"]["forecasts"],
        allow_overwrite=args.overwrite,
    )

    submission_path = export_submission_file(
        forecast_df=forecast_table,
        forecast_date=forecast_date,
        model_key=model_run_id,
        submissions_dir=base_config["paths"]["submissions"],
        allow_overwrite=args.overwrite,
    )

    print("\nWeekly forecast created successfully.\n")

    print(
        forecast_table[
            [
                "forecast_rank",
                "ticker",
                "company",
                "model",
                "predicted_weekly_return",
            ]
        ].to_string(index=False)
    )

    print(f"\nForecast run folder: {run_directory}")
    print(f"Submission-format CSV: {submission_path}")


if __name__ == "__main__":
    main()