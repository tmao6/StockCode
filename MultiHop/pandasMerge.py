import sys


import pandas as pd

import yfinance as yf #for the getStocks
import datetime

from pytrends.request import TrendReq #for the getTrends
from pytrends import dailydata

stockTicker = "VTI"

yesterday = datetime.datetime.now()
manyDay = datetime.timedelta(days=50)
oneDay = datetime.timedelta(days=1)
earlier = yesterday - manyDay
yesterday = yesterday - oneDay


stockData = yf.download(stockTicker, start=str(earlier.date()), end=str(yesterday.date()))
stockData = stockData.drop(columns=['Adj Close'])
export_csv = stockData.to_csv(('data/' + stockTicker + ".csv"), index=True,
                              header=True)  # Don't forget to add '.csv' at

stockData = stockData.rename_axis("date") #needed to merge the two columns



# Login to Google. Only need to run this once, the rest of requests will use the same session.
pytrend = TrendReq()
trendData = dailydata.get_daily_data(stockTicker, (earlier.year), (earlier.month), (yesterday.year),
                                     (yesterday.month))

print(stockData)
print(trendData)

stockAndTrend = pd.concat([stockData, trendData], axis=1,join='inner',sort=False)
export_csv = stockAndTrend.to_csv(('data/' + stockTicker + ".csv"), index=True,
                                  header=True)  # Don't forget to add '.csv' at