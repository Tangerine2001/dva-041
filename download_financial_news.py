import json
import os
import sys
import time

import pandas as pd
from tqdm import tqdm

import reuters_api as reuters


def makeDirs():
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/raw_stock_data"):
        os.mkdir("data/raw_stock_data")


def loadNewsStatus() -> dict:
    if os.path.exists("newsStatus.json"):
        with open("newsStatus.json", "r") as file:
            newsStatus = json.load(file)
    else:
        newsStatus = {}
    return newsStatus


def loadTickerNews(ticker: str) -> pd.DataFrame:
    if os.path.exists(f"data/raw_news_data/{ticker}_news.csv"):
        tickerNews = pd.read_csv(f"data/raw_news_data/{ticker}_news.csv", index_col=0)
    else:
        tickerNews = pd.DataFrame(columns=['Id', 'Date', 'Title', 'Short Description'])
        tickerNews = tickerNews.set_index('Id')
    return tickerNews


def addToTickerNews(tickerNews: pd.DataFrame, article: dict):
    articleId = article['articlesId']
    articleName = article['articlesName']
    articleShortDesc = article['articlesShortDescription']
    articleDate = article['publishedAt']['date'].split()[0]
    tickerNews.loc[articleId] = [articleDate, articleName, articleShortDesc]
    return tickerNews


def downloadFinancialNews(ticker: str):
    makeDirs()
    newsStatus = loadNewsStatus()
    tickerNews = loadTickerNews(ticker)

    limitPerPage = 20
    status = newsStatus.get(ticker, {})
    done = status.get('Done', False)

    with open("symbols.json", "r") as file:
        tickers = json.load(file)
    shortName = tickers[ticker]['shortName']

    # Use one request to get the total number of pages
    data = reuters.getArticlesByKeywordName(shortName, 0, limitPerPage)
    if 'allPages' not in data:
        print(f"API Key limit reached.")
        return
    totalNumPages = data['allPages']

    if not done:
        # Use one request to get the total number of pages
        lastPage = status.get('Last Page', 0)
        for i in tqdm(range(lastPage + 1, totalNumPages)):
            time.sleep(0.3)
            data = reuters.getArticlesByKeywordName(shortName, i, limitPerPage)

            if 'articles' not in data:
                newsStatus[ticker] = {'Done': False, 'Last Page': i}
                break

            articles = data['articles']

            for article in articles:
                if article['articlesId'] in tickerNews.index:
                    continue
                addToTickerNews(tickerNews, article)
        else:
            newsStatus[ticker] = {'Done': True, 'Last Page': lastPage}

    else:
        breakFromArticles = False
        for i in tqdm(range(totalNumPages)):
            time.sleep(0.3)
            data = reuters.getArticlesByKeywordName(shortName, i, limitPerPage)

            if 'articles' not in data:
                break

            articles = data['articles']
            for article in articles:
                if article['articlesId'] in tickerNews.index:
                    breakFromArticles = True
                    break
                addToTickerNews(tickerNews, article)

            if breakFromArticles:
                break
        else:
            newsStatus[ticker] = {'Done': True, 'Last Page': totalNumPages}

    tickerNews = pd.DataFrame(tickerNews)
    tickerNews.to_csv(f"news_data/{ticker}_news.csv")
    with open("newsStatus.json", "w") as file:
        json.dump(newsStatus, file, indent=4)


def main():
    args = sys.argv[1:]
    for tickerName in args:
        print(f"Downloading {tickerName} financial news")
        downloadFinancialNews(tickerName)
        print(f"Downloaded news for {tickerName}\n")


if __name__ == "__main__":
    main()