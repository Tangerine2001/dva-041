import pandas as pd
import numpy as np

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

    def add_sent_score(self):
        self.df["SentScore"] = 0
        return self.df.iloc[19:, :]
