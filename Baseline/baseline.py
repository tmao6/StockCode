



import os
import pandas as pd
import tqdm
from tqdm import tqdm_notebook
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential,load_model
from keras.layers import LSTM, Dense
from keras.layers import Dropout, Activation, Flatten
from keras.optimizers import SGD




df_ge = pd.read_csv( "us.ge.txt", index_col=0)
print(df_ge.head())




'''
plt.figure()
plt.plot(df_ge["Open"])
plt.plot(df_ge["High"])
plt.plot(df_ge["Low"])
plt.plot(df_ge["Close"])
plt.title('GE stock price history')
plt.ylabel('Price (USD)')
plt.xlabel('Days')
plt.legend(['Open','High','Low','Close'], loc='upper left')
plt.show()
'''




'''
plt.figure()
plt.plot(df_ge["Volume"])
plt.title('GE stock volume history')
plt.ylabel('Volume')
plt.xlabel('Days')
plt.show()
'''




print("checking if any null values are present\n", df_ge.isna().sum())




#train_cols = ["Open","High","Low","Close","Volume"]
train_cols = ["Open","High","Low","Close"]
df_train, df_test = train_test_split(df_ge, train_size=0.8, test_size=0.2, shuffle=False)
print("Train and Test size", len(df_train), len(df_test))




# Loading training data into X
x = df_train.loc[:,train_cols].values
# Min Max scalr normalizing
#min_max_scaler = MinMaxScaler()
# Normalizing Training data
#x_train = min_max_scaler.fit_transform(x)
x_train=x
# Normalizing testing
#x_test = min_max_scaler.transform(df_test.loc[:,train_cols])
x_test = df_test.loc[:,train_cols].values
x_test




def build_timeseries(mat, y_col_index, TIME_STEPS):
    # y_col_index is the index of column that would act as output column
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))
    
    for i in tqdm_notebook(range(dim_0)):
        x[i] = mat[i:TIME_STEPS+i]
        y[i] = mat[TIME_STEPS+i, y_col_index]
    print("length of time-series i/o",x.shape,y.shape)
    return x, y




def trim_dataset(mat, batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = mat.shape[0]%batch_size
    if(no_of_rows_drop > 0):
        return mat[:-no_of_rows_drop]
    else:
        return mat




TIME_STEPS = 18
Y_index = 3
X_Train, Y_Train = build_timeseries(x_train, Y_index,TIME_STEPS)
#x_t = trim_dataset(x_t, BATCH_SIZE)
#y_t = trim_dataset(y_t, BATCH_SIZE)
X_Val, Y_Val = build_timeseries(x_test,  Y_index,TIME_STEPS)
#x_val, x_test_t = np.split(trim_dataset(x_temp, BATCH_SIZE),2)
#y_val, y_test_t = np.split(trim_dataset(y_temp, BATCH_SIZE),2)












# Initialising the RNN
model = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(24, return_sequences=True,
               input_shape=(X_Train.shape[1], X_Train.shape[2])))  # returns a sequence of vectors of dimension 64
model.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
model.add(Dropout(0.2))

model.add(Dense(20,activation='relu'))


# Adding the output layer
#model.add(Dense(1, activation="linear"))
model.add(Dense(1))

#model.add(Dense(units = 1))
# Compiling the RNN

model.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=['accuracy'])

history=model.fit(X_Train, Y_Train,
          batch_size=32, epochs=100,
          validation_data=(X_Val, Y_Val))
# Model summary for number of parameters use in the algorithm 
model.summary()




plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'Val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'Val'], loc='upper left')
plt.show()




model.save('my_model_01.hdf5')




#print(model.predict(X_Val)
'''
x_buffer = np.zeros((1,TIME_STEPS,4))
for i in range(X_Val.shape[0]):
    x_buffer[0,:,:] = X_Val[i,:,:]
    print(model.predict( x_buffer), Y_Val[i])
'''




predict = model.predict(X_Val)





plt.figure()
plt.plot(predict)
plt.plot(Y_Val)

plt.title('GE stock price prediction')
plt.ylabel('Price (USD)')
plt.xlabel('Days')
plt.legend(['predict','Actual'], loc='upper left')

plt.savefig("figure.png")
plt.show()











