from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


def download_close_prices(
    tickers: list[str], #e.g. 0700.HK
    start_date: str,    
    output_dir: str,
) -> pd.DataFrame:
    """Downloads daily closing prices for a list of tickers from Yahoo Finance and saves them to CSV.

    Fetches historical daily close data starting from `start_date` up to the current day,
    preserves the requested ticker column order, exports timestamped and latest CSV copies
    to `output_dir`, and returns the processed DataFrame.

    Args:
        tickers (list[str]): List of stock ticker symbols (e.g., ['0700.HK', 'AAPL']).
        start_date (str): Start date for data retrieval in 'YYYY-MM-DD' format.
        output_dir (str): Directory path where the output CSV files will be saved.

    Returns:
        pd.DataFrame: A DataFrame containing daily closing prices indexed by 'Date',
            with columns ordered according to the input `tickers` list.

    Raises:
        RuntimeError: If `yfinance` returns an empty DataFrame (e.g., invalid tickers or 
            no internet connectivity).

    Example:
        from src.hk_equity.data.download import download_close_prices
        df = download_close_prices(
            tickers=['0700.HK', '9988.HK'],
            start_date='2024-01-01',
            output_dir='./'
        )
    """

    #parents = True : If paraenet directory not exist, also create parent directory
    #exist_ok = True: If the path has already exist -> no throw exception
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True) 

    #set the end date to tmr (bc end date is not included)
    end_date = ( 
        datetime.now() + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    raw_data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",      #each row = 1 day
        auto_adjust=False,  #No Auto_adjust
        actions=False,      #Don't download the dividends & Stock Splits
        group_by="column",  #The Multiindex design: [Close then stock name] instead of [stock name -> Close]
        progress=False,     #No show download progress bar
        threads=True,       #Multi-threading
    )

    #Throw exception when nothing exist
    if raw_data.empty:
        raise RuntimeError("No data downloaded from yfinance.")


    close_prices = raw_data["Close"].copy()                 #only get the close price
    close_prices = close_prices.reindex(columns=tickers)    #Sort the stock name with the .ymal given order
    close_prices.index = pd.to_datetime(close_prices.index) #convert the index column to datetime format
    close_prices.index.name = "Date"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    close_prices.to_csv( #Save daily
        output_path / f"daily_close_{timestamp}.csv"
    )

    close_prices.to_csv( #Save latest daily
        output_path / "latest_daily_close.csv"
    )

    return close_prices