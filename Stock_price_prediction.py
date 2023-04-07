# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 17:07:01 2023

@author: Arunim
"""
'''
In this problem we have to predict the closing price of NASDAQ on 23rd December 2021. 
'''
import numpy  as np
import pandas as pd 
import os
for dirname, _, filenames in os.walk('D:/Masters/Stock_price'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#import cudf as pd
#import cupy as cp
'''
import cuml
from cuml import LinearRegression
from cuml.linear_model import LinearRegression
from cuml import Ridge
from cuml.linear_model import Ridge
from cuml.model_selection import train_test_split
from cuml.linear_model import Lasso
from cuml.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
'''

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import plot_model
import pandas as pd
import numpy as np



data = pd.read_csv('D:/Masters/Stock_price/IXIC.csv')
print("Null Value present?", data.isnull().values.any())


'''Visualize the closing price history'''

data_plot = pd.read_csv('D:/Masters/Stock_price/IXIC.csv',sep=',', parse_dates=['Date'],index_col='Date')
import matplotlib.pylab as plt
import math
plt.figure(figsize=(16,8))
plt.plot(data_plot['Close'])
plt.xlabel('Dates',fontsize= 18)
plt.ylabel( 'Closing Price', fontsize = 15)
plt.show()

#creating a new datatframe containing only the close column

df2 = data.filter(['Close'])
df2arr = data_plot['Close'].values
#divide dataset such that 80% goes into training and rest 20% will go for testing
training_data_len = math.ceil( len(df2arr) * 0.8)


#Feature Scaling
sc= MinMaxScaler(feature_range=(0,1))
scaled_data = sc.fit_transform(df2)

#Creating training set and test set
training_set = scaled_data[0:training_data_len,:]


'''LSTM'''


X_train=[]
y_train=[]

for i in range(60,len(training_set)):
    X_train.append(training_set[i-60:i,0])
    y_train.append(training_set[i,0])
X_train,y_train = np.array(X_train), np.array(y_train)

'''Reshape our data. X_train is 2D, LSTM needs 3D'''
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

#Build the LSTM model 
# We add the LSTM layer and later add a few Dropout layers to prevent overfitting. 
# We add the LSTM layer with the following arguments:
# 1. 50 units which is the dimensionality of the output space
# 2. return_sequences=True which determines whether to return the last output in the output sequence, or the full sequence
# 3. input_shape as the shape of our training set.

# When defining the Dropout layers, we specify 0.2, meaning that 20% of the layers will be dropped.
# Thereafter, we add the Dense layer that specifies the output of 1 unit
# After this, we compile our model using the popular adam optimizer and set the loss as the mean_squarred_error.
# Next, we fit the model to run on 100 epochs with a batch size of 32

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))

model.add(LSTM(50, return_sequences=False))

model.add(Dense(25))

model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1)

#test data
X_test=[]
test_data = scaled_data[training_data_len-60:,:]
Y_test=scaled_data[training_data_len:,:]

for i in range(60,len(test_data)):
    X_test.append(test_data[i-60:i,0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

predicted_Stock = model.predict(X_test)

#Evaluating model
from sklearn.metrics import mean_squared_error
MSE=mean_squared_error( Y_test, predicted_Stock)
rmse = math.sqrt(MSE)

#Plotting
#data_Top_eighty = df2arr[:training_data_len]
#data_Bottom_twenty = df2arr[training_data_len:]
predictions = sc.inverse_transform(predicted_Stock)
Y_test_unscaled = sc.inverse_transform(Y_test)




plt.plot(Y_test_unscaled, color = 'red', label = 'Real Stock Price')
plt.plot(predictions, color = 'blue', label = 'Predicted  Stock Price')
plt.title('NASDAQ Stock prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

comparison = np.concatenate([Y_test_unscaled,predictions],axis=1)






    
    











