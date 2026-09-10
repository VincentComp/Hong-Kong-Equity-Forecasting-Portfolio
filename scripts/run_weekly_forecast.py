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


#pass the input
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

    return daily_close


def prepare_portfolio_daily_close(
    daily_close: pd.DataFrame,
    portfolio_tickers: list[str],
) -> pd.DataFrame:
    """Return only the portfolio stocks used for baseline forecasting."""

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

    return daily_close[portfolio_tickers].copy()


def main() -> None:
    """Create, save, and display one model's next-week forecast."""

    args = parse_arguments()

    # Load general project settings and the selected model settings.
    base_config, model_config = load_project_config(
        base_config_path=args.base_config,
        model_config_path=args.model_config,
    )

    # Load the latest saved daily Close-price snapshot.
    daily_close = load_daily_close_prices(
        raw_data_directory=base_config["paths"]["raw_data"],
    )

    portfolio_tickers = list(
        base_config["tickers"].keys()
    )

    # The raw CSV may contain ^HSI. HSI is a benchmark for future advanced
    # features, not a stock to forecast or submit.
    portfolio_daily_close = prepare_portfolio_daily_close(
        daily_close=daily_close,
        portfolio_tickers=portfolio_tickers,
    )

    # Use the requested cutoff, or latest available price date by default.
    data_cutoff = (
        pd.Timestamp(args.as_of)
        if args.as_of is not None
        else portfolio_daily_close.index.max()
    )

    # Build one common context used by every forecasting model.
    context = build_forecast_context(
        daily_close=portfolio_daily_close,
        data_cutoff=data_cutoff,
        weekly_frequency=base_config["forecast"]["weekly_frequency"],
        horizon_trading_days=base_config["forecast"]["horizon_trading_days"],
    )

    # Generate one predicted next-week return per portfolio ticker.
    predicted_return = get_model_forecast(
        weekly_returns=context.weekly_returns,
        model_config=model_config,
    )

    model_key = model_config["model"]["name"]
    model_run_id = model_config["model"]["run_id"]

    model_label = get_model_label(
        model_config=model_config,
    )

    # Convert model output into a standard Project 1 forecast table.
    forecast_table = build_forecast_table(
        predicted_return=predicted_return,
        tickers=base_config["tickers"],
        context=context,
        model_config=model_config,
        model_label=model_label,
    )

    forecast_date = context.data_cutoff.date().isoformat()

    # Keep metadata so the exact data cutoff, model version, and parameters
    # can be reproduced after submission.
    metadata = {
        "project_name": base_config["project_name"],
        "data_source": base_config["data"]["source"],
        "price_field": base_config["data"]["price_field"],
        "auto_adjust": base_config["data"]["auto_adjust"],
        "market_ticker": base_config["market"]["ticker"],
        "weekly_frequency": base_config["forecast"]["weekly_frequency"],
        "horizon_trading_days": (
            base_config["forecast"]["horizon_trading_days"]
        ),
        "model_config_path": args.model_config,
        "model_key": model_key,
        "model_run_id": model_run_id,
        "model_version": model_config["model"]["version"],
        "model_parameters": model_config.get("parameters", {}),
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

    # Save the full research record under:
    # outputs/forecasts/YYYY-MM-DD/model_key/
    run_directory = save_forecast_run(
        forecast_df=forecast_table,
        metadata=metadata,
        forecast_date=forecast_date,
        model_key=model_run_id,
        forecasts_dir=base_config["paths"]["forecasts"],
        allow_overwrite=args.overwrite,
    )

    # Save a clean CSV for review or final Moodle submission.
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