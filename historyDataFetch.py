import yfinance as yf
import pandas as pd

# use yfinance to fetch TSMC stock data
# stock_data = yf.download('2330.TW', start='2010-01-01', end='2023-12-31')
stock_data = yf.download('2330.TW', start='2023-12-01', end='2023-12-31')

# get the required columns and rename the column names
stock_data = stock_data.reset_index()
stock_data = stock_data[['Date', 'High', 'Low', 'Open', 'Close']]
stock_data.columns = ['date', 'high', 'low', 'open', 'close']

# convert the date format to '%Y%m%d'
stock_data['date'] = stock_data['date'].dt.strftime('%Y%m%d')

# sort the data by date
stock_data = stock_data.sort_values(by='date')

# ouput the data to a CSV file
stock_data.to_csv('./tsmc_stock_prices.csv', index=False)