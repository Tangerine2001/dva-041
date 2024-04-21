import pandas as pd
import numpy as np
from download_data import get_news_data
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Define the sentiment analysis model and the labels it'll use:
model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
sentimentlabels = {0: 'neutral', 1: 'positive', 2: 'negative'}


class augment_stockdata():

    def __init__(self, df : pd.DataFrame, closingInd = 3, openInd = 0):
        self.df = df
        self.closingInd = closingInd
        self.openInd = openInd

    def add_ind(self):
        #This method adds technical indicators to a dataframe
        self.df["7MA"] = self.df.iloc[:, self.closingInd].rolling(7).mean()
        self.df["20MA"] = self.df.iloc[:, self.closingInd].rolling(20).mean()
        self.df['MACD'] = self.df.iloc[:, self.closingInd].ewm(span = 26).mean() - self.df.iloc[:, self.openInd].ewm(span = 12, adjust = False).mean()

        self.df['20SD'] = self.df.iloc[:, self.closingInd].rolling(20).std()
        self.df['upper_boll'] = self.df['20MA'] + (self.df['20SD'] * 2)
        self.df['lower_boll'] = self.df['20MA'] - (self.df['20SD'] * 2)

        self.df['EMA'] = self.df.iloc[:, self.closingInd].ewm(com = 0.5).mean()

        self.df['logmom'] = np.log(self.df.iloc[:, self.closingInd] - 1)

        return self.df.iloc[19:, :]

    def add_sent_score(self, ticker: str, sentimentdecay: float = 2):
        newsdata = get_news_data(ticker)

        sentimentdf_as_list = []
        for _, entry in tqdm(list(newsdata.iloc[:10].iterrows()), desc=f'Sentiment scoring for {ticker}'):
            inputs = tokenizer(entry['Title'], return_tensors='pt', padding=True)
            # Determine neutral, positive, and negative sentiment scores for each headline:
            outputs = model(**inputs)[0][0].detach().numpy()
            output_as_dict = {v: outputs[k] for k, v in sentimentlabels.items()}
            output_as_dict['date'] = entry.Date
            sentimentdf_as_list.append(output_as_dict)

        # Output a time series of sentiment scores:
        sentimentdf = pd.DataFrame(sentimentdf_as_list)

        # Pool multiple sentiments in a day by their maximum scores:
        sentimentdf = sentimentdf.groupby(by='date').max()

        # Exponentially tapered-off interpolation of sentiment:
        sentiment_startdate = sentimentdf.index.min()
        timeseries_startdate = self.df.index.min()
        currsentiment = (sentimentdf.loc[sentiment_startdate]
                         if (sentiment_startdate == timeseries_startdate) else np.zeros(3))
        last_sentiment_date = timeseries_startdate
        for date in self.df.index:
            if date not in sentimentdf.index:
                time_since_last_sentiment = (date - last_sentiment_date).days
                self.df.loc[date, sentimentdf.columns] = (currsentiment
                                                          * np.exp(-time_since_last_sentiment / sentimentdecay))
            else:
                currsentiment = sentimentdf.loc[date]
                self.df.loc[date, sentimentdf.columns] = currsentiment
                last_sentiment_date = date

        self.df.sort_index(inplace=True)
        # print(self.df)

        # self.df["SentScore"] = 0
        return self.df.iloc[19:, :]
