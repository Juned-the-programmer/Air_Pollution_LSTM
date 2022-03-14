from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from numpy import array , hstack

# model = load_model("saved_model/Air_Pollution")
# print(model.summary())

dataset_test_ok = pd.read_csv('Dataset/pollution_test_data1.csv')
dataset_test_ok.head()

# integer encode direction
encoder1 = LabelEncoder()
dataset_test_ok.iloc[:,3] = encoder1.fit_transform(dataset_test_ok.iloc[:,3])


# read test data
x1_test = dataset_test_ok['dew'].values
x2_test = dataset_test_ok['temp'].values
x3_test = dataset_test_ok['press'].values
x4_test = dataset_test_ok['wnd_spd'].values
x5_test = dataset_test_ok['wnd_dir'].values
x6_test = dataset_test_ok['snow'].values
x7_test = dataset_test_ok['rain'].values
y_test = dataset_test_ok['pollution'].values # no need to scale


# convert to [rows, columns] structure
x1_test = x1_test.reshape((len(x1_test), 1))
x2_test = x2_test.reshape((len(x2_test), 1))
x3_test = x3_test.reshape((len(x3_test), 1))
x4_test = x4_test.reshape((len(x4_test), 1))
x5_test = x5_test.reshape((len(x5_test), 1))
x6_test = x6_test.reshape((len(x6_test), 1))
x7_test = x7_test.reshape((len(x7_test), 1))
y_test = y_test.reshape((len(y_test), 1))


scaler = MinMaxScaler(feature_range=(0, 1))
x1_test_scaled = scaler.fit_transform(x1_test)
x2_test_scaled = scaler.fit_transform(x2_test)
x3_test_scaled = scaler.fit_transform(x3_test)
x4_test_scaled = scaler.fit_transform(x4_test)
x5_test_scaled = scaler.fit_transform(x5_test)
x6_test_scaled = scaler.fit_transform(x6_test)
x7_test_scaled = scaler.fit_transform(x7_test)


# Step 3 : horizontally stack columns
dataset_test_stacked = hstack((x1_test_scaled,x2_test_scaled,x3_test_scaled,x4_test_scaled,x5_test_scaled,x6_test_scaled,x7_test_scaled))


print ("dataset_stacked.shape" , dataset_test_stacked.shape)




###Prediction#######################################################

dataset_test_X = dataset_test_stacked
print("dataset_test_X :",dataset_test_X.shape)  
test_X_new = dataset_test_X.reshape(1,dataset_test_X.shape[0],dataset_test_X.shape[1])
print(test_X_new.shape)

y_pred = model.predict(test_X_new)

y_pred_inv = scaler.inverse_transform(y_pred)
y_pred_inv = y_pred_inv.reshape(n_steps_out,1)
y_pred_inv = y_pred_inv[:,0]
print("y_pred :",y_pred.shape)
print("y_pred_inv :",y_pred_inv.shape)