from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


def download_close_prices(
    tickers: list[str],
    start_date: str,
    output_dir: str,
) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    end_date = (
        datetime.now() + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    raw_data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        actions=False,
        group_by="column",
        progress=False,
        threads=True,
    )

    if raw_data.empty:
        raise RuntimeError("No data downloaded from yfinance.")

    close_prices = raw_data["Close"].copy()
    close_prices = close_prices.reindex(columns=tickers)
    close_prices.index = pd.to_datetime(close_prices.index)
    close_prices.index.name = "Date"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    close_prices.to_csv(
        output_path / f"daily_close_{timestamp}.csv"
    )

    close_prices.to_csv(
        output_path / "latest_daily_close.csv"
    )

    return close_prices