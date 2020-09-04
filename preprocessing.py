import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    data = df.sort_index(ascending=True,axis=0)
    
    dataset = pd.DataFrame(index=range(0,len(df)), columns=['Close'])
    
    for i in range(0,len(data)):
        dataset["Close"][i] = data["Close"][i]
        
    dataset.index = data.Date
    dataset = dataset.values
    
    train_data = dataset[0:987,:]
    valid_data = dataset[987:,:]
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    
    x_train, y_train = [],[]
    
    for i in range(60,len(train_data)):
        x_train.append(scaled_data[i-60:i,0])
        y_train.append(scaled_data[i,0])
        
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    
    return {
            'dataset' : dataset,
            'valid_data' : valid_data,
            'x_train' : x_train,
            'y_train' : y_train,
            'scaler' : scaler
            }