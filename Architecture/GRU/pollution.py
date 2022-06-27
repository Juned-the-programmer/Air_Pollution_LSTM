# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from pandas import DataFrame , concat
from sklearn.metrics import mean_absolute_error , mean_squared_error
from numpy import mean , concatenate
from math import sqrt
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Activation,GRU,Dropout

from sklearn.preprocessing import LabelEncoder , OneHotEncoder

#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
from numpy import array , hstack
from tensorflow import keras
import tensorflow as tf

# TF_GPU_ALLOCATOR=cuda_malloc_async

dataset = pd.read_csv("../Dataset/LSTM-Multivariate_pollution.csv", header=0, parse_dates=True)

dataset['date'] = pd.to_datetime(dataset['date'])

dataset['year'] = dataset['date'].dt.year
dataset['month'] = dataset['date'].dt.month
dataset['day'] = dataset['date'].dt.day
dataset['hour'] = dataset['date'].dt.hour
print(dataset)

t = dataset.columns.tolist()
dataset = dataset[['dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain','pollution' , 'year' , 'month' , 'day' , 'hour']]
print(dataset)

corr = dataset.corr()
plt.figure(figsize=(20,14))
plot = sns.heatmap(corr , annot=True)
plt.show()

values = dataset.values

groups = [1, 2, 3, 5, 6]
i = 1

# plot each column
plt.figure(figsize=(20,14))
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group], c = "forestgreen")
    plt.title(dataset.columns[group], y=0.75, loc='right', fontsize = 15)
    i += 1
plt.show()

plt.figure(figsize=(20,14))
plt.plot(dataset.pollution[:360] , color='tab:red')
plt.show()


#else slice is invalid for use in labelEncoder
print(dataset.wnd_dir.unique())
dataset= dataset.values
# integer encode direction
encoder = LabelEncoder()
dataset[:,3] = encoder.fit_transform(dataset[:,3])
# #conver to pd.Dataframe else slices error

dataset = pd.DataFrame(dataset)
dataset.columns = ['dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain','pollution' , 'year' , 'month' , 'day' , 'hour']
print(dataset)
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
x_8 = dataset['year'].values
x_9 = dataset['month'].values
x_10 = dataset['day'].values
x_11 = dataset['hour'].values
y = dataset['pollution'].values

#Scatter plot before data transformation
figure , axis = plt.subplots(2,2)
axis[0,0].plot(x_1 , y)
axis[0,0].set_title("Dew vs Pollution")

axis[0,1].plot(x_2 , y)
axis[0,1].set_title("Temp vs Pollution")

axis[1,0].plot(x_3 , y)
axis[1,0].set_title("Press vs Pollution")

axis[1,1].plot(x_4 , y)
axis[1,1].set_title("Wind Speed vs Pollution")

plt.show()

# # Step 1 : convert to [rows, columns] structure
# # x_0 = x_0.reshape((len(x_0) , 1))
x_1 = x_1.reshape((len(x_1), 1))
x_2 = x_2.reshape((len(x_2), 1))
x_3 = x_3.reshape((len(x_3), 1))
x_4 = x_4.reshape((len(x_4), 1))
x_5 = x_5.reshape((len(x_5), 1))
x_6 = x_6.reshape((len(x_6), 1))
x_7 = x_7.reshape((len(x_7), 1))
x_8 = x_8.reshape((len(x_8), 1))
x_9 = x_9.reshape((len(x_9), 1))
x_10 = x_10.reshape((len(x_10), 1))
x_11 = x_11.reshape((len(x_11), 1))
y = y.reshape((len(y), 1))

# Step 2 : normalization 
# x_0_scaled = scaler.fit_transform(x_0)
scaler = MinMaxScaler(feature_range=(0, 1))
x_1_scaled = scaler.fit_transform(x_1)
x_2_scaled = scaler.fit_transform(x_2)
x_3_scaled = scaler.fit_transform(x_3)
x_4_scaled = scaler.fit_transform(x_4)
x_5_scaled = scaler.fit_transform(x_5)
x_6_scaled = scaler.fit_transform(x_6)
x_7_scaled = scaler.fit_transform(x_7)
x_8_scaled = scaler.fit_transform(x_8)
x_9_scaled = scaler.fit_transform(x_9)
x_10_scaled = scaler.fit_transform(x_10)
x_11_scaled = scaler.fit_transform(x_11)
y_scaled = scaler.fit_transform(y)

figure , axis = plt.subplots(2,2)
axis[0,0].plot(x_1_scaled , y_scaled)
axis[0,0].set_title("Dew vs Pollution")

axis[0,1].plot(x_2_scaled , y_scaled)
axis[0,1].set_title("Temp vs Pollution")

axis[1,0].plot(x_3_scaled , y_scaled)
axis[1,0].set_title("Press vs Pollution")

axis[1,1].plot(x_4_scaled , y_scaled)
axis[1,1].set_title("Wind Speed vs Pollution")

plt.show()

plt.figure(figsize=(20,14))
plt.plot(y_scaled[:360] , color='tab:red')
plt.show()


# Step 3 : horizontally stack columns
dataset_stacked = hstack((x_1_scaled, x_2_scaled, x_3_scaled, x_4_scaled,
                          x_5_scaled, x_6_scaled, x_7_scaled, x_8_scaled ,
                          x_9_scaled, x_10_scaled , x_11_scaled, y_scaled))
print ("dataset_stacked.shape" , dataset_stacked.shape)
print(dataset_stacked)

# corr = dataset_stacked.corr()
# plt.figure(figsize=(20,14)) 
# plot = sns.heatmap(corr , annot=True)
# plt.show()

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

# #spliting the dataset--------------------------

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
n_features = 11
#number of features
#n_features = 2

#optimizer learning rate
opt = keras.optimizers.Adam(learning_rate=0.001)
# define model
model = Sequential()
# model.add(Bidirectional(LSTM(50, activation='tanh' , input_shape=(n_steps_in, n_features))))
model.add(GRU(70, activation='tanh' , recurrent_activation='sigmoid' , return_sequences=True , input_shape=(n_steps_in, n_features)))
model.add(Dropout(0.2))
model.add(GRU(70 , activation='tanh' , recurrent_activation='sigmoid' , return_sequences=True ))
model.add(Dropout(0.2))
model.add(GRU(70 , activation='tanh' , recurrent_activation='sigmoid'))
model.add(Dense(n_steps_out))
model.add(Activation('linear'))
model.compile(loss='mae', optimizer=opt , metrics=['accuracy'])

print(model.summary())


# # Fit network #increase the epochs for better model training
history = model.fit(train_X , train_y , epochs=300, batch_size=72 , verbose=2 ,validation_data=(test_X, test_y) ,shuffle=False)
model.save('Air_Pollution_70.h5')

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

y_test_true = test_y

testPredict = model.predict(test_X)
print(testPredict.shape)
testPresict = testPredict.ravel()
print(testPredict.shape)

poll = np.array(dataset["pollution"])
meanop = poll.mean()
stdop = poll.std()
y_test_true = y_test_true*stdop + meanop
testPredict = testPredict*stdop + meanop

plt.figure(figsize=(15,6))
plt.xlim([1000,1250])
plt.ylabel("ppm")
plt.xlabel("hrs")
plt.plot(y_test_true , c="g" , alpha=0.90 , linewidth=2.5)
plt.plot(testPredict , c = "g" , alpha=0.75)
plt.show()

rmse = np.sqrt(mean_squared_error(y_test_true , testPredict))
print("Test (Validation) RMSE = " , rmse)


# 200 epoch => 3.40366628
# 100 epoch => 7.04375111