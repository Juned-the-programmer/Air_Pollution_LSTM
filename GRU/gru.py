# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from pandas import DataFrame , concat
from sklearn.metrics import mean_absolute_error , mean_squared_error
from numpy import mean , concatenate
from math import sqrt
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense,LSTM,Activation,Dropout , Bidirectional , GRU

from sklearn.preprocessing import LabelEncoder , OneHotEncoder

#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
from numpy import array , hstack
from tensorflow import keras
import tensorflow as tf

dataset = pd.read_csv("./Dataset/LSTM-Multivariate_pollution.csv", header=0, parse_dates=True)

t = dataset.columns.tolist()
dataset = dataset[['dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain','pollution']]
print(dataset)

#else slice is invalid for use in labelEncoder
dataset= dataset.values
# integer encode direction
encoder = LabelEncoder()
dataset[:,3] = encoder.fit_transform(dataset[:,3])

#conver to pd.Dataframe else slices error

dataset = pd.DataFrame(dataset)
dataset.columns = ['dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain','pollution']

######## ensure all data is float

#Data Pre-processing step--------------------------------
# x_0 = dataset['date'].values
x_1 = dataset['dew'].values
x_2 = dataset['temp'].values
x_3 = dataset['press'].values
x_4 = dataset['wnd_spd'].values
x_5 = dataset['wnd_dir'].values
x_6 = dataset['snow'].values
x_7 = dataset['rain'].values
y = dataset['pollution'].values
#x_1 = x_1.values
#x_2 = x_2.values
#y = y.values


# Step 1 : convert to [rows, columns] structure
# x_0 = x_0.reshape((len(x_0) , 1))
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

# dataset_stacked = hstack((x_0 , x_1, x_2, x_3, x_4,
#                           x_5, x_6, x_7, y))
# print(dataset_stacked)

# Step 2 : normalization 
scaler = MinMaxScaler(feature_range=(0, 1))
# x_0_scaled = scaler.fit_transform(x_0)
x_1_scaled = scaler.fit_transform(x_1)
x_2_scaled = scaler.fit_transform(x_2)
x_3_scaled = scaler.fit_transform(x_3)
x_4_scaled = scaler.fit_transform(x_4)
x_5_scaled = scaler.fit_transform(x_5)
x_6_scaled = scaler.fit_transform(x_6)
x_7_scaled = scaler.fit_transform(x_7)
y_scaled = scaler.fit_transform(y)

# Step 3 : horizontally stack columns
dataset_stacked = hstack((x_1_scaled, x_2_scaled, x_3_scaled, x_4_scaled,
                          x_5_scaled, x_6_scaled, x_7_scaled, y_scaled))
print ("dataset_stacked.shape" , dataset_stacked.shape)
print(dataset_stacked)

#1. n_steps_in : Specify how much data we want to look back for prediction
#2. n_step_out : Specify how much multi-step data we want to forecast

# split a multivariate sequence into samples
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
n_steps_in, n_steps_out = 360 , 24
# covert into input/output
X, y = split_sequences(dataset_stacked, n_steps_in, n_steps_out)
print ("X.shape" , X.shape) 
print ("y.shape" , y.shape)

#spliting the dataset--------------------------

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
train_X, test_X,train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 0)
#split_point = 1258*25
#train_X , train_y = X[:split_point, :] , y[:split_point, :]
#test_X , test_y = X[split_point:, :] , y[split_point:, :]


print(train_X.shape) #[n_datasets,n_steps_in,n_features]
print(train_y.shape) #[n_datasets,n_steps_out]
print(test_X.shape) 
print(test_y.shape) 
n_features = 7
#number of features
#n_features = 2

model = Sequential()
# model.add(Bidirectional(LSTM(50, activation='tanh' , input_shape=(n_steps_in, n_features))))
model.add(GRU(70, activation='tanh' , recurrent_activation='sigmoid' , return_sequences=True , input_shape=(n_steps_in, n_features)))
model.add(Dropout(0.2))
model.add(GRU(70 , activation='tanh' , recurrent_activation='sigmoid' , return_sequences=True ))
model.add(Dropout(0.2))
model.add(GRU(70 , activation='tanh' , recurrent_activation='sigmoid'))
model.add(Dense(n_steps_out))
model.add(Activation('linear'))
model.compile(loss='mae', optimizer='adam' , metrics=['accuracy'])

# fit network
history = model.fit(train_X, train_y, epochs=300, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
model.save("Air_Pollution_output_24/Air_Pollution.h5")

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()