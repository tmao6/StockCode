import sys

import os
import json
import time
import math
import matplotlib.pyplot as plt
import pandas as pd

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


def plot_results_multiple(predicted_data, true_data, prediction_len, ticker, isTrends):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
	# Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
    if isTrends:
        plt.title(str(ticker)+" with Trends")
        plt.savefig(str(ticker)+"_with_Trends.png")
    else:
        plt.title(str(ticker)+" without Trends")
        plt.savefig(str(ticker)+"_without_Trends.png")


    plt.show()


def main():
    stockTickerList = ["VOX", "VCR", "VDC", "VDE", "VFH", "VHT", "VIS", "VGT", "VAW", "VNQ", "VPU"]


    for stockTicker in stockTickerList:
        isTrends = True
        getStockData = True
        getTrendData = True

        #MAKE SURE BOTH DATA SET EXACT SAME LENGTH AND FILLED

        yesterday = datetime.datetime.now()
        manyDay = datetime.timedelta(days=10*365)
        oneDay = datetime.timedelta(days=1)
        earlier = yesterday - manyDay
        yesterday = yesterday - oneDay

        if isTrends:
            configs = json.load(open('configWithTrends_looping.json', 'r'))
        else:
            configs = json.load(open('configWithoutTrends.json', 'r'))


        if getStockData:
            stockData = yf.download(stockTicker, start=str(earlier.date()), end=str(yesterday.date()))
            stockData = stockData.drop(columns=['Adj Close'])
            export_csv = stockData.to_csv(('data/' + stockTicker + ".csv"), index=True, header=True)  # Don't forget to add '.csv' at

        if getTrendData:
            # Login to Google. Only need to run this once, the rest of requests will use the same session.
            pytrend = TrendReq()
            trendData0 = dailydata.get_daily_data(stockTicker, (earlier.year), (earlier.month), (yesterday.year), (yesterday.month))
            trendData1 = dailydata.get_daily_data(stockTicker+" stock", (earlier.year), (earlier.month), (yesterday.year), (yesterday.month))
            trendData2 = dailydata.get_daily_data(stockTicker+" price", (earlier.year), (earlier.month), (yesterday.year), (yesterday.month))
            trendData4 = dailydata.get_daily_data(stockTicker+" vanguard", (earlier.year), (earlier.month), (yesterday.year), (yesterday.month))
            trendData5 = dailydata.get_daily_data("Dow Jones", (earlier.year), (earlier.month), (yesterday.year), (yesterday.month))
            trendData6 = dailydata.get_daily_data("Trump", (earlier.year), (earlier.month), (yesterday.year), (yesterday.month))
            stockAndTrend = pd.concat([stockData, trendData0, trendData1, trendData2, trendData4, trendData5, trendData6], axis=1, join='inner', sort=False)
            export_csv = stockAndTrend.to_csv(('data/' + stockTicker + ".csv"), index=True, header=True)  # Don't forget to add '.csv' at

        #
        #     configs['data']['columns'] = ['Close', 'Open', 'High', 'Low', 'Volume',
        #                                   stockTicker, stockTicker+" price",
        #                                   stockTicker+" vanguard", "Dow Jones", "Trump"]
        #
        #
        # # configs = json.load(open('config_noVolume.json', 'r'))
        # if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
        #
        # data = DataLoader(
        #     ("data/"+stockTicker+".csv"),
        #     configs['data']['train_test_split'],
        #     configs['data']['columns']
        # )
        #
        # model = Model()
        # model.build_model(configs)
        # x, y = data.get_train_data(
        #     seq_len=configs['data']['sequence_length'],
        #     normalise=configs['data']['normalise']
        # )
        #
        # # # out-of memory generative training
        # # steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
        # # model.train_generator(
        # #     data_gen=data.generate_train_batch(
        # #         seq_len=configs['data']['sequence_length'],
        # #         batch_size=configs['training']['batch_size'],
        # #         normalise=configs['data']['normalise']
        # #     ),
        # #     epochs=configs['training']['epochs'],
        # #     batch_size=configs['training']['batch_size'],
        # #     steps_per_epoch=steps_per_epoch,
        # #     save_dir=configs['model']['save_dir']
        # # )
        #
        # # in-memory training
        # model.train(x, y, epochs=configs['training']['epochs'], batch_size=configs['training']['batch_size'],
        #             save_dir=configs['model']['save_dir'])
        #
        # x_test, y_test = data.get_test_data(
        #     seq_len=configs['data']['sequence_length'],
        #     normalise=configs['data']['normalise']
        # )
        #
        # predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])
        # # predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
        # # predictions = model.predict_point_by_point(x_test)
        #
        # plot_results_multiple(predictions, y_test, configs['data']['sequence_length'], stockTicker, isTrends)
        # # plot_results(predictions, y_test)


if __name__ == '__main__':
    main()