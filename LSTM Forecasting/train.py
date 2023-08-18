import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

#load data
def parse(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

dataset = pd.read_csv('../Dataset/LSTM-Multivariate_pollution.csv' ,  index_col=0 , parse_dates =True)

dataset.columns = ['pollution' ,'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'

print(dataset)

values = dataset.values
print(values)

print(values[0])
print(values[:,0])

#Plot the graph
groups = [0,1,2,3,4,5,6,7]
i = 1
plt.figure()

for group in groups:
    plt.subplot(len(groups), 1 , i)
    plt.plot(values[:,group])
    plt.title(dataset.columns[group] , y=0.5 , loc='right')
    i += 1

plt.show()

#Series To Supervised
def series_to_supervised(data , n_in=1 , n_out=1 , dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols , names = list() , list()

    for i in range(n_in,0,-1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    for i in range(0,n_out):
        cols.append(df.shift(-i))
        if  i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    agg = pd.concat(cols , axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# Encoding of Data
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])

print(values[:24])

values = values.astype('float32')
print(values[:12])

#Scaling of Data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(values)

print(scaled[:12])

reframed = series_to_supervised(scaled , 1 , 1)
print(reframed[:12])

reframed.drop(reframed.columns[[9,10,11,12,13,14,15]] , axis=1 , inplace=True)
print(reframed[:12])

#Define and Fit model 
values = reframed.values
n_train_hours = 365*24
train = values[:n_train_hours,:]
test = values[n_train_hours:,:]

print(train)
print(train.shape)
print(test.shape)

#Split the data into input and output
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

print(train_X.shape)
print(test_X.shape)

print(test_X)
print(test_y)


# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

#Creating the model
from keras.models import Sequential
from keras.layers import LSTM,Dense

model = Sequential()
model.add(LSTM(50 , input_shape = (train_X.shape[1] , train_X.shape[2]))) 
model.add(Dense(1))
model.compile(loss='mae' , optimizer='adam' , metrics=['accuracy'])

#Fit the network
history = model.fit(train_X , train_y , epochs=50 , batch_size=72 , validation_data=(test_X , test_y) , verbose=2 , shuffle=False)
model.save('saved_model/Air_Pollution.h5')

#Prediction 
y_what = model.predict(test_X)
print(y_what)

test_X = test_X.reshape((test_X.shape[0] , test_X.shape[2]))
print(test_X)

print(test_X.shape)
print(test_X[:,1:])
print(test_X[:,1:].shape)

from numpy import concatenate
inv_yhat = concatenate((y_what, test_X[:, 1:]), axis=1)
print(inv_yhat)
inv_ywhat = scaler.inverse_transform(inv_yhat)
print(inv_ywhat)