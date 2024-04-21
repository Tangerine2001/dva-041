import torch
import pickle
import datetime
import yfinance as yf
import augment_data
from models import lstm_predictor
import numpy as np
import plotUtil

prediction_period = 23
predict_step = 3

class forecastor():

    def __init__(self, tickers, out, today = True, start_date = None):
        self.tickers = tickers
        self.outfile= out

    def inference(self, ticker):
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
        with open('model_cache\scalerSENT_' + ticker + '.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        scaled_data = scaler.transform(df)
        print(scaled_data.shape)
        num_feats = scaled_data.shape[1]
        input = torch.tensor(scaled_data[np.newaxis, -1 * (prediction_period - predict_step) :, :])

        print(input.shape)
        
        input_days = [(today_date - datetime.timedelta(days = i - predict_step)).strftime("%m/%d/%Y") for i in range(prediction_period, 0, -1)]


        model = lstm_predictor(prediction_period - predict_step, num_feats, 15, 32, predict_step)
        model.load_state_dict(torch.load("model_cache\predictorSENTAUG_" + ticker))

        forecast = scaler.inverse_transform(np.broadcast_to(model(input).detach().numpy().reshape(-1, 1), (3, num_feats)))[:, 3]
        print(forecast)
        #save forecast in csv
        #plot forecast
        plot = plotUtil.plotter("Price Forecasts for " + ticker, "Input Prices", "Date", "Price", input_prices, input_days[:-1 * predict_step])
        plot.add_predicted("Predicted Price", "orange", forecast, input_days[-1 * predict_step :])
        plot.get_plot(verbose= True)
        return None
    
f = forecastor(["GOOG"], None)
f.inference("GOOG")