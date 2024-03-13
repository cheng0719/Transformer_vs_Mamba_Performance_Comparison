import yfinance as yf
import pandas as pd

# 使用YFinance抓取台積電股票資料
stock_data = yf.download('2330.TW', start='2023-01-01', end='2023-12-31')

# 取出所需欄位並重新命名欄位名稱
stock_data = stock_data.reset_index()
stock_data = stock_data[['Date', 'High', 'Low', 'Open', 'Close']]
stock_data.columns = ['date', 'high', 'low', 'open', 'close']

# 將數值的最小位設為整數個位數
stock_data['high'] = stock_data['high'].astype(int)
stock_data['low'] = stock_data['low'].astype(int)
stock_data['open'] = stock_data['open'].astype(int)
stock_data['close'] = stock_data['close'].astype(int)

# 轉換日期格式為'%Y%m%d'
stock_data['date'] = stock_data['date'].dt.strftime('%Y%m%d')

# 依照日期遞增排列
stock_data = stock_data.sort_values(by='date')

# 將資料輸出至CSV檔案
stock_data.to_csv('./tsmc_stock_prices_testing_INT.csv', index=False)
