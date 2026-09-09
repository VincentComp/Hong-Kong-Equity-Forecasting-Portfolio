"""Generate and save a general Project 1 weekly-return forecast.

Examples:
    python scripts/run_weekly_forecast.py
    python scripts/run_weekly_forecast.py --as-of 2026-09-18
    python scripts/run_weekly_forecast.py --as-of 2026-09-18 --model ewma
    python scripts/run_weekly_forecast.py --as-of 2026-09-18 --lookback 8

Run this after the last trading day of a completed week. The script is general:
it detects the latest completed Friday-ending week in your saved price data and
forecasts the following Friday-ending week. No course dates are hard-coded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.hk_equity.data.preprocess import calculate_returns, to_weekly_close
from src.hk_equity.utils.io import export_submission_file, save_forecast_run


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser( #create a parser
        description="Create a weekly-return forecast from the latest saved Close prices."
    )
    parser.add_argument( #choose the as-of-date
        "--as-of",
        default=None,
        help="Data cutoff date in YYYY-MM-DD. Default: latest date in latest_daily_close.csv.",
    )
    parser.add_argument( #choose the model
        "--model",
        choices=["zero", "moving_average", "ewma"],
        default="moving_average",
        help="Forecast method. Default: moving_average.",
    )
    parser.add_argument( #choose the lookback for moving preiod
        "--lookback",
        type=int,
        default=4,
        help="Completed weeks used by moving_average or EWMA. Default: 4.",
    )
    return parser.parse_args()


def calculate_next_week_forecast(#get the forecast result
    completed_weekly_returns: pd.DataFrame,
    model_name: str,
    lookback: int,
) -> tuple[pd.Series, str]:
    """Forecast the NEXT week using all returns known at the latest completed week.

    This is intentionally different from a backtest forecast table, which is
    shifted by one row to align historical prediction with its realised week.
    For a live next-week forecast, the latest completed weekly return is known
    and must be included in the lookback window.
    """
    if len(completed_weekly_returns) < lookback:
        raise ValueError(f"Need at least {lookback} completed weekly returns.")

    if model_name == "zero":
        return pd.Series(0.0, index=completed_weekly_returns.columns), "Zero Return Baseline"

    if model_name == "moving_average":
        return (
            completed_weekly_returns.tail(lookback).mean(),
            f"{lookback}-Week Moving Average",
        )

    return (
        completed_weekly_returns.ewm(span=lookback, adjust=False).mean().iloc[-1],
        f"{lookback}-Week EWMA",
    )


def create_forecast_table(
    daily_close: pd.DataFrame,
    tickers: dict[str, str],
    model_name: str,
    lookback: int,
    data_cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Build one Project 1 forecast row per stock from a historical data snapshot."""
   
    usable_daily_close = daily_close.loc[:data_cutoff].dropna(how="all")                #get the date with date ≤ as-of-date
    if usable_daily_close.empty:                                                        #if nothing then exception
        raise ValueError("No prices exist on or before the requested --as-of date.")

    latest_data_date = usable_daily_close.index.max()                                   #get the latest valid date


    weekly_close = to_weekly_close(usable_daily_close)

    # A partial current week must never be treated as a completed forecast week.
    weekly_close = weekly_close.loc[weekly_close.index <= data_cutoff.normalize()]
    weekly_returns = calculate_returns(weekly_close).dropna(how="all")
    if weekly_returns.empty:
        raise ValueError("No complete weekly returns could be calculated.")

    latest_completed_week = weekly_returns.index.max()
    completed_returns = weekly_returns.loc[:latest_completed_week]
    predicted_return, model_label = calculate_next_week_forecast(
        completed_weekly_returns=completed_returns,
        model_name=model_name,
        lookback=lookback,
    )

    latest_close = weekly_close.loc[latest_completed_week]
    target_week_end = latest_completed_week + pd.Timedelta(days=7)
    target_week_start = target_week_end - pd.Timedelta(days=4)

    forecast_table = pd.DataFrame({
        "forecast_date": data_cutoff.date().isoformat(),
        "data_last_available_date": latest_data_date.date().isoformat(),
        "last_completed_week_end": latest_completed_week.date().isoformat(),
        "target_week_start": target_week_start.date().isoformat(),
        "target_week_end": target_week_end.date().isoformat(),
        "ticker": list(tickers.keys()),
        "company": list(tickers.values()),
        "model": model_label,
        "lookback_weeks": lookback if model_name != "zero" else 0,
        "latest_weekly_close_hkd": latest_close.reindex(tickers.keys()).values,
        "predicted_weekly_return": predicted_return.reindex(tickers.keys()).values,
    })

    if forecast_table["predicted_weekly_return"].isna().any():
        missing = forecast_table.loc[
            forecast_table["predicted_weekly_return"].isna(), "ticker"
        ].tolist()
        raise ValueError(f"Missing forecast value(s) for: {missing}")

    forecast_table = forecast_table.sort_values(
        "predicted_weekly_return", ascending=False
    ).reset_index(drop=True)
    forecast_table.insert(0, "forecast_rank", np.arange(1, len(forecast_table) + 1))
    return forecast_table, latest_data_date


def main() -> None:
    args = parse_arguments() #get the args

    with open("configs/base.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    daily_close_path = Path(config["paths"]["raw_data"]) / "latest_daily_close.csv"
    daily_close = pd.read_csv(daily_close_path, index_col="Date", parse_dates=True).sort_index()

    data_cutoff = pd.Timestamp(args.as_of) if args.as_of else daily_close.index.max()
    forecast_table, last_data_date = create_forecast_table(
        daily_close=daily_close,
        tickers=config["tickers"],
        model_name=args.model,
        lookback=args.lookback,
        data_cutoff=data_cutoff,
    )

    forecast_date = data_cutoff.date().isoformat()
    metadata = {
        "data_source": config["data"]["source"],
        "price_field": config["data"]["price_field"],
        "auto_adjust": config["data"]["auto_adjust"],
        "model_key": args.model,
        "lookback_weeks": args.lookback,
        "requested_data_cutoff": forecast_date,
        "actual_last_available_data_date": last_data_date.date().isoformat(),
        "forecast_target": "Next Friday-ending weekly stock return",
        "number_of_stocks": len(forecast_table),
    }

    #save the csv under forecast and submissions directory
    run_dir = save_forecast_run(
        forecast_df=forecast_table,
        metadata=metadata,
        forecast_date=forecast_date,
        forecasts_dir=config["paths"]["forecasts"],
    )
    submission_path = export_submission_file(
        forecast_df=forecast_table,
        forecast_date=forecast_date,
        submissions_dir=config["paths"]["submissions"],
    )

    print("Weekly forecast created successfully.\n")
    print(forecast_table[[
        "forecast_rank", "ticker", "company", "predicted_weekly_return"
    ]].to_string(index=False))
    print(f"\nForecast run folder: {run_dir}")
    print(f"Submission-format CSV: {submission_path}")


if __name__ == "__main__":
    main()