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
from sklearn.preprocessing import LabelEncoder , OneHotEncoder

df = pd.read_csv("Dataset/LSTM-Multivariate_pollution.csv")
print(df.head())

plt.figure(figsize = (15,8))
ax = sns.heatmap(df.corr(),annot=True, fmt="1.1f")
plt.show()

sns.pairplot(df)
plt.show()

plt.plot('pollution', data=df)
plt.show()

# Data Configuration for training model 
t = df.columns.tolist()
print(t)
df = df[['dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain','pollution']]
print(df)

#else slice is invalid for use in labelEncoder
df = df.values
# integer encode direction
encoder = LabelEncoder()
df[:,3] = encoder.fit_transform(df[:,3])

#conver to pd.Dataframe else slices error

df = pd.DataFrame(df)
df.columns = ['dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain','pollution']

x_1 = df['dew'].values
x_2 = df['temp'].values
x_3 = df['press'].values
x_4 = df['wnd_dir'].values
x_5 = df['wnd_spd'].values
x_6 = df['snow'].values
x_7 = df['rain'].values
y = df['pollution'].values
#x_1 = x_1.values
#x_2 = x_2.values
#y = y.values


# Step 1 : convert to [rows, columns] structure
x_1 = x_1.reshape((len(x_1), 1))
x_2 = x_2.reshape((len(x_2), 1))
x_3 = x_3.reshape((len(x_3), 1))
x_4 = x_4.reshape((len(x_4), 1))
x_5 = x_5.reshape((len(x_5), 1))
x_6 = x_6.reshape((len(x_6), 1))
x_7 = x_7.reshape((len(x_7), 1))
y = y.reshape((len(y), 1))
print ("x_1.shape" , x_1.shape) 
print ("x_2.shape" , x_2.shape) 
print ("y.shape" , y.shape)

# Step 2 : normalization 
scaler = MinMaxScaler(feature_range=(0, 1))
x_1_scaled = scaler.fit_transform(x_1)
x_2_scaled = scaler.fit_transform(x_2)
x_3_scaled = scaler.fit_transform(x_3)
x_4_scaled = scaler.fit_transform(x_4)
x_5_scaled = scaler.fit_transform(x_5)
x_6_scaled = scaler.fit_transform(x_6)
x_7_scaled = scaler.fit_transform(x_7)
y_scaled = scaler.fit_transform(y)

# Step 3 : horizontally stack columns
dataset_stacked = hstack((x_1_scaled, x_2_scaled,x_2_scaled, x_3_scaled,
                          x_4_scaled, x_5_scaled,x_7_scaled,y_scaled))
print ("dataset_stacked.shape" , dataset_stacked.shape)

def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# choose a number of time steps #change this accordingly
n_steps_in, n_steps_out = 60 , 30
# covert into input/output
X, y = split_sequences(dataset_stacked, n_steps_in, n_steps_out)
print ("X.shape" , X.shape) 
print ("y.shape" , y.shape)

def custom_ts_multi_data_prep(dataset, target, start, end, window, horizon):
    X = []
    y = []
    start = start + window
    if end is None:
        end = len(dataset) - horizon
    for i in range(start, end):
        indices = range(i-window, i)
        X.append(dataset[indices])
        indicey = range(i+1, i+1+horizon)
        y.append(target[indicey])
    return np.array(X), np.array(y)

hist_window = 10
horizon = 10
TRAIN_SPLIT = 35040
x_train, y_train = custom_ts_multi_data_prep(X, y,0,TRAIN_SPLIT, hist_window, horizon)
x_vali, y_vali = custom_ts_multi_data_prep(X, y, TRAIN_SPLIT, None, hist_window, horizon)

print ('Multiple window of past history\n')
print (x_train[0])
print ('\n Target horizon\n')
print (y_train[0]) 

batch_size = 256
buffer_size = 150
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().shuffle(buffer_size).batch(batch_size).repeat()
val_data = tf.data.Dataset.from_tensor_slices((x_vali, y_vali))

lstm_model = tf.keras.models.Sequential([tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(200, return_sequences=True),input_shape=x_train.shape[-2:]),
    tf.keras.layers.Dense(20, activation='tanh'),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)),
    tf.keras.layers.Dense(20, activation='tanh'),
    tf.keras.layers.Dense(20, activation='tanh'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(units=horizon),
])
lstm_model.compile(optimizer='adam', loss='mse')
print(lstm_model.summary())

model_path = 'Bidirectional_LSTM_Multivariate.h5'
early_stopings = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min')
checkpoint =  tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0)
callbacks=[early_stopings,checkpoint] 
history = lstm_model.fit(train_data,epochs=150,steps_per_epoch=100,validation_data=val_data,validation_steps=50,verbose=1)