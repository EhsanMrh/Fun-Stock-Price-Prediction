# Import libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout 

from sklearn.preprocessing import MinMaxScaler

# Import and read dataset
dataset = pd.read_csv('NSE-Tata-Global-Beverages-Limited')
dataset.head()