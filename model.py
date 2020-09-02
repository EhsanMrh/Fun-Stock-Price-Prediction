# Import libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Import and read dataset
dataset = pd.read_csv('dataset/NSE-Tata-Global-Beverages-Limited.csv')
dataset.head(5)

