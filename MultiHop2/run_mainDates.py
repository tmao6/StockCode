import sys

import os
import json
import time
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter


import yfinance as yf #for the getStocks
import datetime

from pytrends.request import TrendReq #for the getTrends
from pytrends import dailydata

from core.data_processor import DataLoader
from core.model import Model

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len, ticker, isTrends, filename, split):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)


    dataframe = pd.read_csv(filename)
    i_split = int(len(dataframe) * split) + prediction_len

    dates = mdates.date2num(pd.to_datetime(dataframe.iloc[i_split:len(dataframe), 0]))

	# Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot_date(dates[0:(i+1) * prediction_len], np.transpose(np.array(padding + data)), label='Prediction', fmt="-")


    months = mdates.MonthLocator()  # every month
    months_fmt = mdates.DateFormatter('%Y-%m')

    plt.plot_date(dates, true_data, fmt="-", color="cornflowerblue", linewidth="0.5")

    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(months_fmt)

    # ax.plot((np.array(dataframe.iloc[i_split:len(dataframe), 0])),true_data)

    # format the coords message box
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax.grid(True)

    fig.autofmt_xdate()


    if isTrends:
        plt.title(str(ticker)+" with Trends")
        plt.savefig(str(ticker)+"_with_Trends.png")
        print("saved")
    else:
        plt.title(str(ticker)+" without Trends")
        plt.savefig(str(ticker)+"_without_Trends.png")
        print("saved")


    plt.show()


def main():
    stockTicker = "VOX"
    isTrends = True

    getStockData = False
    getTrendData = False

    #MAKE SURE BOTH DATA SET EXACT SAME LENGTH AND FILLED

    yesterday = datetime.datetime.now()
    manyDay = datetime.timedelta(days=7*365)
    oneDay = datetime.timedelta(days=1)
    earlier = yesterday - manyDay
    yesterday = yesterday - oneDay

    if getStockData:
        stockData = yf.download(stockTicker, start=str(earlier.date()), end=str(yesterday.date()))
        stockData = stockData.drop(columns=['Adj Close'])
        export_csv = stockData.to_csv(('data/' + stockTicker + ".csv"), index=True, header=True)  # Don't forget to add '.csv' at

    if getTrendData:
        # Login to Google. Only need to run this once, the rest of requests will use the same session.
        pytrend = TrendReq()
        trendData = dailydata.get_daily_data(stockTicker+" stock", (earlier.year), (earlier.month), (yesterday.year), (yesterday.month))
        stockAndTrend = pd.concat([stockData, trendData], axis=1, join='inner', sort=False)
        export_csv = stockAndTrend.to_csv(('data/' + stockTicker + ".csv"), index=True, header=True)  # Don't forget to add '.csv' at

    if isTrends:
        configs = json.load(open('configWithTrends.json', 'r'))
    else:
        configs = json.load(open('configWithoutTrends.json', 'r'))


    # configs = json.load(open('config_noVolume.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    data = DataLoader(
        ("data/"+stockTicker+".csv"),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )


    model = Model()
    model.build_model(configs)
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    # # out-of memory generative training
    # steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    # model.train_generator(
    #     data_gen=data.generate_train_batch(
    #         seq_len=configs['data']['sequence_length'],
    #         batch_size=configs['training']['batch_size'],
    #         normalise=configs['data']['normalise']
    #     ),
    #     epochs=configs['training']['epochs'],
    #     batch_size=configs['training']['batch_size'],
    #     steps_per_epoch=steps_per_epoch,
    #     save_dir=configs['model']['save_dir']
    # )

    # in-memory training
    model.train(x, y, epochs=configs['training']['epochs'], batch_size=configs['training']['batch_size'],
                save_dir=configs['model']['save_dir'])

    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    print("x_test")
    print(x_test)
    print("-----")
    print("y_test")
    print(y_test)

    predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])


    # predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    # predictions = model.predict_point_by_point(x_test)

    plot_results_multiple(predictions, y_test, configs['data']['sequence_length'], stockTicker, isTrends, 'data/' + stockTicker + ".csv", configs['data']['train_test_split'])
    # plot_results(predictions, y_test)


if __name__ == '__main__':
    main()