from flask import Flask, render_template, request
import pandas as pd
import plotly.graph_objs as go
import os

app = Flask(__name__)

# Generate or load dummy stock data
def load_or_generate_dummy_data():
    if not os.path.exists('forecasts.csv'):
        generate_dummy_data()
    df = pd.read_csv('forecasts.csv')
    df['Date'] = pd.to_datetime(df['Date'])  # Convert 'Date' column to Timestamp objects
    return df

def generate_dummy_data():
    date_range = pd.date_range(start='2024-04-01', end='2024-04-30')
    num_days = len(date_range)
    
    data = {
        'Date': date_range,
        'AAPL': [100 + i for i in range(num_days)],
        'GOOGL': [200 + i for i in range(num_days)],
        'MSFT': [150 + i for i in range(num_days)],
        'AMZN': [250 + i for i in range(num_days)],  # Tech sector
        'JPM': [170 + i for i in range(num_days)],  # Tech sector
    }
    df = pd.DataFrame(data)
    df.to_csv('forecasts.csv', index=False)


@app.route('/')
def index():
    df = load_or_generate_dummy_data()
    available_stocks = df.columns[1:].tolist()  # Exclude the 'Date' column
    return render_template('index.html', available_stocks=available_stocks)

@app.route('/plot', methods=['POST'])
def plot():
    df = load_or_generate_dummy_data()
    selected_stocks = request.form.getlist('stocks')
    print("Selected Stocks:", selected_stocks)

    # Extracting selected historical data option
    historical_data = request.form.get('historical')
    print("Selected Historical Data Option:", historical_data)

    # Filtering data based on selected historical data option
    if historical_data == 'prev_week':
        print("Filtering data for previous week...")
        df_filtered = df[df['Date'] >= pd.Timestamp.today() - pd.Timedelta(days=7)]
        df_filtered = df_filtered[df['Date'] <= pd.Timestamp.today()]
    elif historical_data == 'prev_month':
        print("Filtering data for previous month...")
        df_filtered = df[df['Date'] >= pd.Timestamp.today() - pd.Timedelta(days=30)]
        df_filtered = df_filtered[df['Date'] <= pd.Timestamp.today()]
    elif historical_data == 'prev_year':
        print("Filtering data for previous year...")
        df_filtered = df[df['Date'] >= pd.Timestamp.today() - pd.Timedelta(days=365)]
        df_filtered = df_filtered[df['Date'] <= pd.Timestamp.today()]
    else: # For "All Time", no need to filter the data
        df_filtered = df

    print("Filtered DataFrame:")
    print(df.head())

    traces = []
    for stock in selected_stocks:
        stock = stock + '_perc_change'
        trace = go.Scatter(x=df_filtered['Date'], y=df_filtered[stock], mode='lines', name=stock)
        traces.append(trace)

    layout = go.Layout(title='Stock Forecasted Percent Change', xaxis=dict(title='Date'), yaxis=dict(title='Percent Change'))
    fig = go.Figure(data=traces, layout=layout)
    plot_div = fig.to_html(full_html=False)

    return render_template('plot.html', plot_div=plot_div)


if __name__ == '__main__':
    app.run(debug=True)