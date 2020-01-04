import yfinance as yf
import datetime

stockTicker = "VTI"

yesterday = datetime.datetime.now()
DD1 = datetime.timedelta(days=5*365)
DD2 = datetime.timedelta(days=1)
earlier = yesterday - DD1
yesterday = yesterday - DD2

df = yf.download(stockTicker, start=str(earlier.date()) , end=str(yesterday.date()))

df = df.drop(columns=['Adj Close'])

export_csv = df.to_csv(('data/'+stockTicker+".csv"), index = True, header=True) #Don't forget to add '.csv' at the end of the path

