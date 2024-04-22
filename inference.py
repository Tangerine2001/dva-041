import torch
import pickle
import datetime
import yfinance as yf
import augment_data
from models import lstm_predictor
import numpy as np
import plotUtil
import csv

prediction_period = 23
predict_step = 3

class forecastor():

    def __init__(self, tickers, out = "forecasts.csv", today = True, start_date = None):
        self.tickers = tickers
        self.outfile= out
        self.forecast_dict = {}

    def inference(self, ticker, show_plots = False):
        #get data
        today_date = datetime.datetime.today()

        delta = datetime.timedelta(days= 70)
        starting_date = today_date - delta

        start_data = str(starting_date).split()[0]
            
                
        stock_data = yf.download([ticker], start= start_data)
        input_prices = stock_data["Close"].values[-1 * (prediction_period - predict_step) :]

        #augment data

        aug = augment_data.augment_stockdata(stock_data)
        aug.add_ind()
        df = aug.add_sent_score(ticker)

        #load predictor and scaler
        with open('model_cache/scalerSENT_' + ticker + '.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        scaled_data = scaler.transform(df)
        print(scaled_data.shape)
        num_feats = scaled_data.shape[1]
        input = torch.tensor(scaled_data[np.newaxis, -1 * (prediction_period - predict_step) :, :])

        print(input.shape)
        
        input_days = [(today_date - datetime.timedelta(days = i - predict_step)).strftime("%m/%d/%Y") for i in range(prediction_period, 0, -1)]


        model = lstm_predictor(prediction_period - predict_step, num_feats, 15, 32, predict_step)
        model.load_state_dict(torch.load("model_cache/predictorSENTAUG_" + ticker))

        forecast = scaler.inverse_transform(np.broadcast_to(model(input).detach().numpy().reshape(-1, 1), (3, num_feats)))[:, 3]
        print(forecast)
        #save forecast in csv
        if len(self.forecast_dict.keys()) == 0:
            for day_ind in range(len(input_days)):
                if day_ind < prediction_period - predict_step:
                    self.forecast_dict[input_days[day_ind]] = {ticker : input_prices[day_ind]}
                else:
                    self.forecast_dict[input_days[day_ind]] = {ticker : forecast[day_ind - (prediction_period - predict_step)]}
                if day_ind == 0:
                    self.forecast_dict[input_days[day_ind]][ticker + "_perc_change"] = 0.0
                else:
                    prev = self.forecast_dict[input_days[day_ind - 1]][ticker]
                    self.forecast_dict[input_days[day_ind]][ticker + "_perc_change"] = (self.forecast_dict[input_days[day_ind]][ticker] - prev) / prev
        else:
            for day_ind in range(len(input_days)):
                if day_ind < prediction_period - predict_step:
                    self.forecast_dict[input_days[day_ind]][ticker] = input_prices[day_ind]
                else:
                    self.forecast_dict[input_days[day_ind]][ticker] = forecast[day_ind - (prediction_period - predict_step)]
                if day_ind == 0:
                    self.forecast_dict[input_days[day_ind]][ticker + "_perc_change"] = 0.0
                else:
                    prev = self.forecast_dict[input_days[day_ind - 1]][ticker]
                    self.forecast_dict[input_days[day_ind]][ticker + "_perc_change"] = (self.forecast_dict[input_days[day_ind]][ticker] - prev) / prev
        #plot forecast
        plot = plotUtil.plotter("Price Forecasts for " + ticker, "Input Prices", "Date", "Price", input_prices, input_days[:-1 * predict_step])
        plot.add_predicted("Predicted Price", "orange", forecast, input_days[-1 * predict_step :])
        if show_plots:
            plot.get_plot(verbose= True)

    def save_csv(self):
        with open(self.outfile, 'w', newline='') as f:
            fieldnames = ["Date"]
            fieldnames.extend(self.tickers)
            fieldnames.extend([ticker + "_perc_change" for ticker in self.tickers])
            writer = csv.DictWriter(f, fieldnames= fieldnames)
            writer.writeheader()
            for day in self.forecast_dict.keys():
                row = {'Date' : day}
                for ticker in self.tickers:
                    row[ticker] = self.forecast_dict[day][ticker]
                    row[ticker + "_perc_change"] = self.forecast_dict[day][ticker + "_perc_change"]
                writer.writerow(row)
        f.close()

    def gen_csv(self, show_plots):
        for stock in self.tickers:
            self.inference(stock, show_plots = show_plots)
        self.save_csv()


#change this to generate forecasts for more stocks
STOCK_LIST = ["AAPL", "ADBE", "AMZN", "BAC", "C", "GOOGL", "GS", "IBM", "JPM", "LYFT", "META", "MS", "MSFT", "NFLX", "NVDA", "QCOM", "T", "TSLA", "UBER", "VZ", "WFC"]


f = forecastor(STOCK_LIST)
f.gen_csv(show_plots = False)