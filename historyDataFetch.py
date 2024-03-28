import yfinance as yf
import pandas as pd

# Fetch TSMC stock data using YFinance
stock_data = yf.download('2330.TW', start='2024-01-01', end='2024-2-23')

# Extract the required columns and rename the column names
stock_data = stock_data.reset_index()
# stock_data = stock_data[['Date', 'High', 'Low', 'Open', 'Close']]
# stock_data.columns = ['date', 'high', 'low', 'open', 'close']
stock_data = stock_data[['Date', 'Open']]
stock_data.columns = ['date', 'open']

# Set the minimum digit of the numerical value to an integer digit
# stock_data['high'] = stock_data['high'].astype(int)
# stock_data['low'] = stock_data['low'].astype(int)
stock_data['open'] = stock_data['open'].astype(int)
# stock_data['close'] = stock_data['close'].astype(int)

# Convert the date format to "%Y-%m-%d %H:%M:%S"
stock_data['date'] = stock_data['date'].dt.strftime("%Y-%m-%d %H:%M:%S")

# Sort in ascending order by date
stock_data = stock_data.sort_values(by='date')

# Output the data to a CSV file
stock_data.to_csv('./dataset/tsmc_stock_prices_INT_open_only_datetime_backtesting.csv', index=False)
