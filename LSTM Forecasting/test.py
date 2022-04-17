from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from numpy import array , hstack

#load model
model = load_model("saved_model/Air_Pollution.h5")
print(model.summary())

#load data
dataset = pd.read_csv('test.csv' , index_col=0 , parse_dates =True)

dataset.columns = ['pollution' ,'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'

print(dataset)

values = dataset.values
print(values)

print(values[0])
print(values[:,0])

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

#split the data into input and output
values = reframed.values
test = values[:8761,:]
test_X, test_y = test[:, :-1], test[:, -1]
print(test_X.shape)

#Reshape into 3D
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(test_X.shape)

#Prediction
prediction = model.predict(test_X)
print(prediction)

test_X = test_X.reshape((test_X.shape[0] , test_X.shape[2]))
print(test_X)

print(test_X.shape)
print(test_X[:,1:])
print(test_X[:,1:].shape)

from numpy import concatenate
inv_ywhat = concatenate((prediction , test_X[:,1:]) , axis=1)
print(inv_ywhat)
inv_what = scaler.inverse_transform(inv_ywhat)
print(inv_what)