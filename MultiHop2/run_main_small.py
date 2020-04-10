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

import datetime #datetime for configuring which dates to extract from getStocks

import time

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
    '''
    
    configs = json.load(open('configWithTrendsSmall.json', 'r'))
    
    #Creates a new DataLoader object, see core/data_processor to see what it does.
    data = DataLoader(
        configs['data']['filename'],
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    #Creates model save directory
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

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
     
    stockTicker = "VOX"
    plot_results_multiple(predictions, y_test, configs['data']['sequence_length'], stockTicker, True, configs['data']['filename'], configs['data']['train_test_split'])
    # plot_results(predictions, y_test)


if __name__ == '__main__':
    main()