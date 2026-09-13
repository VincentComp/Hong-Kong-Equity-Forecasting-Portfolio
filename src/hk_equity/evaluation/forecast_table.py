from __future__ import annotations

import numpy as np
import pandas as pd

from src.hk_equity.data.context import ForecastContext


def build_forecast_table(
    predicted_return: pd.Series,
    tickers: dict[str, str],
    context: ForecastContext,
    model_config: dict,
    model_label: str,
    source_model_profile: dict[str, str] | None = None,
    source_model_run_id: dict[str, str] | None = None,
    source_model_label: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build one standardised Project 1 forecast table.

    Every model must use this same output format. This ensures that Moving
    Average, EWMA, Ridge, LightGBM, LSTM, and future ensemble models can all
    produce consistent CSV files for evaluation and Moodle submission.

    Args:
        predicted_return: One forecasted next-week return per ticker.
        tickers: Mapping of ticker symbol to company name.
        context: Shared data/date information for this forecast run.
        model_config: YAML model configuration dictionary.
        model_label: Readable model name for outputs.

    Returns:
        DataFrame with one ranked forecast row per stock.
    """

    ticker_list = list(tickers.keys())

    # Ensure all required target stocks have one forecast value.
    predicted_return = predicted_return.reindex(ticker_list)

    if predicted_return.isna().any():
        missing_tickers = predicted_return[
            predicted_return.isna()
        ].index.tolist()

        raise ValueError(
            "Missing predicted weekly return for: "
            f"{', '.join(missing_tickers)}"
        )

    latest_weekly_close = context.weekly_close.loc[
        context.latest_completed_week
    ].reindex(ticker_list)

    model_name = model_config["model"]["name"]
    model_version = model_config["model"]["version"]

    forecast_table = pd.DataFrame({
        "forecast_date": context.data_cutoff.date().isoformat(),
        "data_last_available_date": (
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
        "ticker": ticker_list,
        "company": [tickers[ticker] for ticker in ticker_list],
        "model": model_label,
        "model_key": model_name,
        "model_version": model_version,
        "latest_weekly_close_hkd": latest_weekly_close.values,
        "predicted_weekly_return": predicted_return.values,
    })

    if source_model_profile is not None:
        forecast_table["source_model_profile"] = [
            source_model_profile.get(
                ticker,
                model_name,
            )
            for ticker in ticker_list
        ]

    if source_model_run_id is not None:
        forecast_table["source_model_run_id"] = [
            source_model_run_id.get(
                ticker,
                model_config["model"]["run_id"],
            )
            for ticker in ticker_list
        ]

    if source_model_label is not None:
        forecast_table["source_model_label"] = [
            source_model_label.get(
                ticker,
                model_label,
            )
            for ticker in ticker_list
        ]

    # Add all model parameters as output columns for auditability.
    # Example: parameter_lookback_weeks = 4
    for parameter_name, parameter_value in model_config.get(
        "parameters",
        {},
    ).items():
        forecast_table[f"parameter_{parameter_name}"] = parameter_value

    forecast_table = forecast_table.sort_values(
        by="predicted_weekly_return",
        ascending=False,
    ).reset_index(drop=True)

    forecast_table.insert(
        loc=0,
        column="forecast_rank",
        value=np.arange(1, len(forecast_table) + 1),
    )

    return forecast_table