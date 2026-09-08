from pathlib import Path

import pandas as pd
import yaml

from src.hk_equity.data.preprocess import (
    to_weekly_close,
    calculate_returns,
)
from src.hk_equity.models.baseline import (
    zero_return_forecast,
    moving_average_forecast,
    ewma_forecast,
)
from src.hk_equity.evaluation.metrics import (
    forecast_metrics,
)


def main():
    with open("configs/base.yaml", "r") as file:
        config = yaml.safe_load(file)

    daily_close = pd.read_csv(
        "data/raw/latest_daily_close.csv",
        index_col="Date",
        parse_dates=True,
    )

    weekly_close = to_weekly_close(daily_close)
    weekly_returns = calculate_returns(weekly_close)

    forecasts = {
        "Zero Return": zero_return_forecast(weekly_returns),
        "4-Week Moving Average": moving_average_forecast(
            weekly_returns,
            lookback_weeks=4,
        ),
        "4-Week EWMA": ewma_forecast(
            weekly_returns,
            span_weeks=4,
        ),
    }

    test_start = config["periods"]["test_start"]
    test_end = config["periods"]["test_end"]

    weekly_test = weekly_returns.loc[test_start:test_end]

    result_rows = []

    for model_name, forecast_table in forecasts.items():
        forecast_test = forecast_table.loc[test_start:test_end]

        for ticker in weekly_returns.columns:
            metrics = forecast_metrics(
                actual=weekly_test[ticker],
                predicted=forecast_test[ticker],
            )

            result_rows.append({
                "model": model_name,
                "ticker": ticker,
                **metrics,
            })

    results = pd.DataFrame(result_rows)

    output_dir = Path(config["paths"]["backtests"])
    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        output_dir / "baseline_metrics.csv",
        index=False,
    )

    print(
        results
        .sort_values(["model", "MAE"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()