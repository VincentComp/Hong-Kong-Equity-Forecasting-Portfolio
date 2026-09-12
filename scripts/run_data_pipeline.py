"""
python -m scripts.run_data_pipeline
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.hk_equity.data.download import download_close_prices


CONFIG_PATH = Path("configs/base.yaml")


def main() -> None:
    """Download portfolio stocks and the configured benchmark index."""

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    portfolio_tickers = list(config["tickers"].keys())
    benchmark_ticker = config["market"]["ticker"]

    if benchmark_ticker in portfolio_tickers:
        raise ValueError(
            "The benchmark ticker must not also be a portfolio ticker: "
            f"{benchmark_ticker}"
        )

    download_tickers = [
        *portfolio_tickers,
        benchmark_ticker,
    ]

    close_prices = download_close_prices(
        tickers=download_tickers,
        start_date=config["data"]["start_date"],
        output_dir=config["paths"]["raw_data"],
    )

    missing_columns = [
        ticker
        for ticker in download_tickers
        if ticker not in close_prices.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "The downloaded data is missing these requested columns: "
            f"{missing_columns}"
        )

    print("Download completed successfully.")
    print(f"Portfolio tickers downloaded: {len(portfolio_tickers)}")
    print(f"Benchmark downloaded: {benchmark_ticker}")
    print(f"Columns: {close_prices.columns.tolist()}")
    print(close_prices.tail())


if __name__ == "__main__":
    main()