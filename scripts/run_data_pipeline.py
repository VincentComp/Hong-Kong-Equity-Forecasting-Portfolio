from pathlib import Path

import yaml

from src.hk_equity.data.download import download_close_prices


CONFIG_PATH = Path("configs/base.yaml") #Note: this is using default setting


def main():
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file) #load the yaml file

    tickers = list(config["tickers"].keys())

    close_prices = download_close_prices(
        tickers=tickers,
        start_date=config["data"]["start_date"],
        output_dir=config["paths"]["raw_data"],
    )

    print("Download completed.")
    print(close_prices.tail())


if __name__ == "__main__":
    main()