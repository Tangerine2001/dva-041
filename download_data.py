import json
import os.path

import pandas as pd
import yfinance as yf
import os
from tqdm import tqdm

# Some of the tickers are different between our news data file names and what we input into the yfinance API, put them
# here:
TICKER_CORRECTION_DICT = {'GOOG': 'GOOGL'}

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def get_data(ticker: str) -> pd.DataFrame:
    data = yf.download(ticker, progress=False)
    return data

def get_news_data(ticker: str, path_to_newsdata: str = 'news_data') -> pd.DataFrame:
    return pd.read_csv(os.path.join(
        path_to_newsdata, TICKER_CORRECTION_DICT.get(ticker, ticker) + '_news.csv')
        , parse_dates=['Date'])

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