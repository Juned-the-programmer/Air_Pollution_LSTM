from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from numpy import array , hstack

model = load_model("saved_model(74,24)/Air_Pollution.h5")
print(model.summary())

dataset_test_ok = pd.read_csv('Dataset/new_test_data.csv')
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

# split a multivariate sequence into samples
# def split_sequences(sequences, n_steps_in, n_steps_out):
#     X, y = list(), list()
#     for i in range(len(sequences)):
#     # find the end of this pattern
#         end_ix = i + n_steps_in
#         out_end_ix = end_ix + n_steps_out-1
#     # check if we are beyond the dataset
#         if out_end_ix > len(sequences):
#             break
#         # gather input and output parts of the pattern
#         seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
#         X.append(seq_x)
#         y.append(seq_y)
#     return array(X), array(y)

# # choose a number of time steps #change this accordingly
# n_steps_in, n_steps_out = 74 , 24
# # covert into input/output
# X, y = split_sequences(dataset_test_stacked, n_steps_in, n_steps_out)
# print ("X.shape" , X.shape) 
# print ("y.shape" , y.shape)

###Prediction#######################################################

dataset_test_X = dataset_test_stacked
print("dataset_test_X :",dataset_test_X.shape) 
print(dataset_test_X.shape[0])
print(dataset_test_X.shape[1]) 
test_X_new = dataset_test_X.reshape(1,dataset_test_X.shape[0],dataset_test_X.shape[1])

y_pred = model.predict(test_X_new)
print("Prediction of the data")
print(y_pred)

y_pred_inv = scaler.inverse_transform([[y_pred[0][2]]])
print("Inverse of the prediction data")
print(y_pred_inv*100)   
# y_pred_inv = y_pred_inv.reshape(24,1)
# # print(y_pred_inv)
# y_pred_inv = y_pred_inv[:,0]
# # print(y_pred_inv)
# print("y_pred :",y_pred.shape)
# print("y_pred_inv :",y_pred_inv.shape)  