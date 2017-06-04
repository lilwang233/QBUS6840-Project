#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 18:22:40 2017

@author: Max
"""

import pandas as pd
import datetime as dt
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler

from keras.layers.core import Dense 
from keras.models import Sequential

#%% Function used to set dates for plotting
def perdelta(start, end, delta):
    curr = start
    while curr < end:
        yield curr
        curr += delta

#%% Read data
file = 'ele_card_total.csv'

df = pd.read_csv(file)

df['Date'] = [df['Date'][i].replace('M','/') for i in range(len(df['Date']))]

df['Date'] = [dt.datetime.strptime(date, '%Y/%m').date() for date in df['Date']]

dates = list(df['Date'])
values = list(df['Total'])
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df = df.Total

ax = df.plot(title = 'Electronic Payment Over Time')
ax.set_ylabel('Amount ($millions)')

#%% Multiplicative Decomposition
res = sm.tsa.seasonal_decompose(df,model='multiplicative')
resplot = res.plot()

x = np.array(range(6,len(df)-6))

seasonal_index = np.array(res.seasonal[3:15])

y = np.array(res.trend)
y = y[~np.isnan(y)]
trend_fit = np.poly1d(np.polyfit(x,y, deg = 1))

pred_val = np.array(range(len(df), len(df)+6))

pred_trend = trend_fit(pred_val)

pred = np.multiply(pred_trend,seasonal_index[3:9])

date = []

end = dt.date(2017,9,1)
current = dt.date(2017, 4, 1)    

# Generate dates
while current<= end:
    date.append(current)
    current += relativedelta(months=1)
    

xall = np.multiply(trend_fit(x),res.seasonal[6:-6])

plt.figure()
plt.plot(dates, values, label = 'Observed')
plt.plot(dates[6:-6], xall, label = 'Predicted')
plt.ylabel('Amount in Millions')
plt.xlabel('Time')
plt.legend(loc='upper left')
plt.title('Multiplicative Decomposition Forecasts')

#%% Naive
pred = []
for i in range(6):
    pred.append(df[len(df)+i-12])
    
plt.figure()
plt.plot(dates, values, label = 'Observed')
plt.plot(date,pred, label = 'Predicted')
plt.ylabel('Amount in Millions')
plt.xlabel('Time')
plt.legend(loc='upper left')
plt.title('Seasonal Naive Forecasts')


#%% ARIMA

# Define period
M = 12

# Log data
lg_data = df.copy()
lg_data = np.log(df)
lg_data = lg_data[6:]
lg_data_d = np.diff(lg_data)

fig, axes = plt.subplots(figsize=(12,9),nrows=2, ncols=1)
ax = df[6:].plot(title = 'Monthly Electronic Payments',ax=axes[0])
ax.set_ylabel('Amount ($millions)')

ax = lg_data.plot(title = 'Log Monthly Electronic Payments',ax=axes[1])
ax.set_ylabel('Log Amount ($millions)')

# Seasonality differencing
lg_data_ds = lg_data[12:].values - lg_data[:-12].values
# first order differencing
lg_data_dsd = np.diff(lg_data_ds)

fig = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(111)
ax1.plot(lg_data_ds)
plt.title('Seasonally Differenced Electronic Payment')
ax2 = fig.add_subplot(111)
ax2.plot(lg_data_dsd)
plt.title('Regular Difference of the Seasonal Differenced')

# ACF & PACF plots
fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(lg_data, lags=40, ax=ax1)
ax1.set_title("ACF: Log Data")
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(lg_data, lags=40, ax=ax2)
ax2.set_title("PACF: Log Data")

fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(lg_data_d, lags=40, ax=ax1)
ax1.set_title("ACF: First Order Difference Log Data")
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(lg_data_d, lags=40, ax=ax2)
ax2.set_title("PACF: First Order Difference Log Data")

fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(lg_data_ds, lags=40, ax=ax1)
ax1.set_title("ACF: Seasonal Difference Log Data")
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(lg_data_ds, lags=40, ax=ax2)  
ax2.set_title("PACF: Seasonal Difference Log Data")   

fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(lg_data_dsd, lags=40, ax=ax1)
ax1.set_title("ACF: First Order Difference of Seasonal Difference")
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(lg_data_dsd, lags=40, ax=ax2)  
ax2.set_title("PACF: First Order Difference of Seasonal Difference") 

# Declare model
sarima_model = sm.tsa.statespace.SARIMAX(lg_data, order=(2,1,1), seasonal_order=(2,1,1,12))  

result = sarima_model.fit(disp=False)

# Forecasting
forecasts = result.forecast(6)

# Display forecasting
plt.figure()
plt.plot(dates, df.values, label = 'Observed')
plt.plot(dates[13:],np.exp(result.fittedvalues[13:]), label = 'Predicted')
plt.ylabel('Amount in Millions')
plt.xlabel('Time')
plt.legend(loc='upper left')
plt.title('Seasonal ARIMA Forecasts')

#%% NNT

data = df.values.astype('float32')

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data.reshape(-1,1))

# prepare training data
time_lag = 12
Xtrain, Ytrain = [], []
for i in range(23 - time_lag, len(data) - time_lag - 1):
    frame = np.append(data[i:(i+time_lag)].T,data[i+time_lag - 23])
    Xtrain.append(frame)   # pick up the section in time_window size
    Ytrain.append(data[i+time_lag:(i+time_lag + 1)].T)       # pick up the next one as the prediction
Xtrain = np.array(Xtrain).reshape(len(Xtrain),time_lag+1)    # Convert them from list to array   
Ytrain = np.array(Ytrain).reshape(len(Ytrain),1)

# Setup neural net
model = Sequential()
model.add(Dense(50, input_dim=time_lag+1, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Training
model.fit(Xtrain, Ytrain, epochs=100, batch_size=2, verbose=2, validation_split=0.05)


# Predicting
# make predictions
pred = []
for i in range(6):
    if (-time_lag+i) < 0:
        testdata = np.append(data[-time_lag+i:],np.array(pred))
    else:
        testdata = np.array(pred[-time_lag+i:])
    testdata = np.append(testdata,data[-11+i])
    testPredict = model.predict(testdata.reshape((1,-1)))
    pred.append(testPredict)
testPredict = scaler.inverse_transform(np.array(pred).reshape(1,-1))

# Plot results
plt.figure()
plt.plot(dates, df.values, label = 'Observed')
plt.plot(date,testPredict.T, label = 'Predicted')
plt.ylabel('Amount in Millions')
plt.xlabel('Time')
plt.legend(loc='upper left')
plt.title('Neural Net Forecasts')
