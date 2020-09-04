# Import libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Import dataset
df=pd.read_csv("dataset/NSE-TATA.csv")
df.head()

df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
df.index=df['Date']

plt.figure(figsize=(16,8))
plt.plot(df["Close"],label='Close Price history')

# Preprocessing datas
from preprocessing import preprocess_data
pro_data = preprocess_data(df)

# Creat a Long Short Term Memory NN
lstm_model = Sequential()
lstm_model.add(LSTM(units = 50, 
                    return_sequences = True, 
                    input_shape = (pro_data['x_train'].shape[1], 1)))
lstm_model.add(LSTM(units = 50))
lstm_model.add(Dense(1))


inputs_data = pro_data['dataset'][len(pro_data['dataset'])-len(pro_data['valid_data']) - 60 : ]
inputs_data = inputs_data.reshape(-1,1)
inputs_data = pro_data['scaler'].transform(inputs_data)

lstm_model.compile(loss = keras.losses.mean_squared_error, 
                   optimizer = 'adam')
lstm_model.fit(pro_data['x_train'],
               pro_data['y_train'],
               epochs = 1,
               batch_size = 1,
               verbose = 2)

# Testing the LSTM model
X_test=[]
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_closing_price = lstm_model.predict(X_test)
predicted_closing_price = pro_data['scaler'].inverse_transform(predicted_closing_price)

# Saving the model
lstm_model.save('lstm_model.h5')