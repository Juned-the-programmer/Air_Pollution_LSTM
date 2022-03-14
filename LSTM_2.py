import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

from numpy import array , hstack

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.recurrent import LSTM
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras import metrics

df = pd.read_csv("Dataset/LSTM-Multivariate_pollution.csv")
print(df.head())

plt.figure(figsize = (15,8))
ax = sns.heatmap(df.corr(),annot=True, fmt="1.1f")
plt.show()