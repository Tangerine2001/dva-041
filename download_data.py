import json
import os.path

import pandas as pd
import yfinance as yf
from tqdm import tqdm


def get_data(ticker: str) -> pd.DataFrame:
    data = yf.download(ticker, progress=False)
    return data


def download_tickers():
    with open("tickers.json", "r") as file:
        tickers = json.load(file)

    if not os.path.exists("data"):
        os.mkdir("data")

    for ticker in tqdm(tickers.keys()):
        data = get_data(ticker)
        data.to_csv(f"data/{ticker}.csv")


if __name__ == "__main__":
    download_tickers()
