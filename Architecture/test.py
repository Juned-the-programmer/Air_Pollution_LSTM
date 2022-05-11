from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder , MinMaxScaler
from numpy import hstack , array

model = load_model("pollution/Air_Pollution.h5")
print(model.summary())

dataset = pd.read_csv("test.csv" , header=0 , parse_dates=True)

dataset['date'] = pd.to_datetime(dataset['date'])

dataset['year'] = dataset['date'].dt.year
dataset['month'] = dataset['date'].dt.month
dataset['day'] = dataset['date'].dt.day
dataset['hour'] = dataset['date'].dt.hour

t = dataset.columns.tolist()
dataset = dataset[['dew' , 'temp' , 'press' , 'wnd_dir' , 'wnd_spd' , 'snow' , 'rain' , 'pollution' , 'year' , 'month' , 'day' , 'hour']]

dataset = dataset.values 
encoder = LabelEncoder()
dataset[:,3] = encoder.fit_transform(dataset[:,3])

dataset = pd.DataFrame(dataset)
dataset.columns = ['dew' , 'temp' , 'press' , 'wnd_dir' , 'wnd_spd' , 'snow' , 'rain' , 'pollution' , 'year' , 'month' , 'day' , 'hour']
print(dataset)

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
scaler = MinMaxScaler(feature_range=(0, 1))
# x_0_scaled = scaler.fit_transform(x_0)
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

# Step 3 : horizontally stack columns
dataset_stacked = hstack((x_1_scaled, x_2_scaled, x_3_scaled, x_4_scaled,
                          x_5_scaled, x_6_scaled, x_7_scaled, x_8_scaled ,
                          x_9_scaled, x_10_scaled , x_11_scaled))
print ("dataset_stacked.shape" , dataset_stacked.shape)
print(dataset_stacked)

dataset_test_X = dataset_stacked
print("dataset_test_X :",dataset_test_X.shape) 
print(dataset_test_X.shape[0])
print(dataset_test_X.shape[1]) 
test_X_new = dataset_test_X.reshape(1,dataset_test_X.shape[0],dataset_test_X.shape[1])

y_pred = model.predict(test_X_new)
print("Prediction of the data")
y_pred_inv = scaler.inverse_transform(y_pred)
y_pred_inv = y_pred_inv.ravel()
print(y_pred_inv)