'''
Unused imports
'''
import sys #used to make pauses to code
import time #not sure (could use to time the program)
import math #not sure
from matplotlib.dates import DateFormatter
'''
Imports
'''
import os #used for setting that lets tensor flow
import json #used in the config files:
            #confinWithoutTrends.json and configWithTrends.json

import matplotlib.pyplot as plt #used for plotting
import pandas as pd #used for databases
import numpy as np #used for calculations

import matplotlib.dates as mdates #dates for plotting/ estimating

import yfinance as yf #for the getStocks section (downloads yahoo stock data)
import datetime #datetime for configuring which dates to extract from getStocks

from pytrends.request import TrendReq #for the getTrends section (downloads Google trends data)
from pytrends import dailydata #trick to get long term normalized data from Google trends

from core.data_processor import DataLoader #no error, from the core folder
from core.model import Model

os.environ['KMP_DUPLICATE_LIB_OK']='True' #setting that lets tensorflow run

#below not really used, seems to be for single prediction (not multiple)
def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show() #


def plot_results_multiple(predicted_data, true_data, prediction_len, ticker, isTrends, filename, split):
    '''
    Plots results from multiple predictions
    '''


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
    #MAKE SURE BOTH DATASETS (yahoo stock and google trends) EXACT SAME LENGTH AND FILLED
    '''
    Runs main code. TODO: Make it into a function that inputs: "Ticker", "Dates of Interest", "Trendword 1", "Trendword 2", etc...

    Inputs: None
    Outputs: Plot with stock fluctuations as a percent change from the start of window

    Key modifiable parameters:
    stockTicker: Which stock we are predicting?
    isTrends: Are we predicting using Google Trends or simply from previous price
        (normally True, false only for debug/ comparison)
    getStockData: Do we need to download a new Yahoo stock dataset or have we downloaded already?
    getTrendData: Do we need to download a new Google Trends dataset or have we downloaded already?
    '''

    stockTicker = "VOX"

    isTrends = True

    getStockData = True
    getTrendData = True

    '''
    Sets dates of interest that are used to extract data. 
    Data is downloaded from "earlier" to "yesterday"
    
    Key modifiable parameters:
    manyDay = datetime.timedelta(days=7*365): sets how many days we want to look backwards for training and test datasets
    
    Key parameters: 
    today: gets todays date for data analysis
    '''
    today = datetime.datetime.now()
    manyDay = datetime.timedelta(days=7*365)
    oneDay = datetime.timedelta(days=1)
    earlier = today - manyDay
    yesterday = today - oneDay

    '''
    Two IF statements that are used to decide whether or not to get Yahoo and Google Trends data or not
    
    Outputs: Saves csv data with data columns used for learning
    '''
   #TODO: make it try if the dataset exists. if it try then pass, else catch and download

    if getStockData:
        stockData = yf.download(stockTicker, start=str(earlier.date()), end=str(yesterday.date()))
        stockData = stockData.drop(columns=['Adj Close'])
        export_csv = stockData.to_csv(('data/' + stockTicker + ".csv"), index=True, header=True)  # Don't forget to add '.csv' at

    if getTrendData:
        # Login to Google. Only need to run this once, the rest of requests will use the same session.
        pytrend = TrendReq()
        trendData1 = dailydata.get_daily_data("coronavirus", (earlier.year), (earlier.month), (yesterday.year), (yesterday.month))
        trendData2 = dailydata.get_daily_data("coronavirus", (earlier.year), (earlier.month), (yesterday.year), (yesterday.month))
        trendData3 = dailydata.get_daily_data("coronavirus", (earlier.year), (earlier.month), (yesterday.year), (yesterday.month))

        stockAndTrend = pd.concat([stockData, trendData1, trendData2, trendData3], axis=1, join='inner', sort=False) #Combines the stockData and trendData datasets
        export_csv = stockAndTrend.to_csv(('data/' + stockTicker + ".csv"), index=True, header=True)  # Don't forget to add '.csv' at


    #Two IF statements open correct config file based on whether or not isTrends is on
    if isTrends:
        configs = json.load(open('configWithTrends.json', 'r'))
    else:
        configs = json.load(open('configWithoutTrends.json', 'r'))


    # configs = json.load(open('config_noVolume.json', 'r')) #left over from original file

    #Creates model save directory
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])


    #TODO: Make the "configs['data']['train_test_split']" and "configs['data']['columns']" inputs to the exe, not hardcoded in the JSON file
    #Creates a new DataLoader object, see core/data_processor to see what it does.
    data = DataLoader(
        ("data/"+stockTicker+".csv"), #passes the filename for the stockTicker
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    #TODO: Make the "configs['data']['sequence_length']" and "configs['data']['normalise']" inputs to the exe, not hardcoded in the JSON file
    #Creates a new Model object, see core/model to see what it does.
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