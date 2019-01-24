#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 16:49:31 2017

@author: Chloe
"""

import pandas as pd
import datetime as dt
import statsmodels.api as sm
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from keras.layers.core import Dense 
from keras.models import Sequential

#%% Sesonal Naive Method
def baseline(df):
    pred = []
    for i in range(6):
        pred.append(df[len(df)+i-12])
    return pred

#%% Multiplicative decomposition
def Multiplicative(df):
    # Decompose data
    res = sm.tsa.seasonal_decompose(df,model='multiplicative')
    
    # Fit first order polynomial for trend
    x = np.array(range(6,len(df)-6))
    y = np.array(res.trend)
    y = y[~np.isnan(y)]
    trend_fit = np.poly1d(np.polyfit(x, y, deg = 1))
    
    # Predict trend component
    pred_val = np.array(range(len(df), len(df)+6))
    pred_trend = trend_fit(pred_val)
    
    # Make prediction using seasonal index and trend
    pred = np.multiply(pred_trend,res.seasonal[pred_val - 12])
    return pred

#%% Seasonal Arima
def sarima(df):
    # Log data
    lg_data = df.copy()
    lg_data = np.log(df)
    
    # Define model
    sarima_model = sm.tsa.statespace.SARIMAX(lg_data, order=(2,1,1), seasonal_order=(2,1,1,12))  
    
    result = sarima_model.fit(disp=False)
    
    # Forecasting
    forecasts = result.forecast(6)
    forecasts = np.exp(forecasts)
    return forecasts

#%% Neural Network
def NN(data, time_lag, neurons):
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data.reshape(-1,1))
    
    # prepare data for training
    Xtrain, Ytrain = [], []
    for i in range(23 - time_lag, len(data) - time_lag - 1):
        frame = np.append(data[i:(i+time_lag)].T,data[i+time_lag - 23]) # use the previous seasona's value
        Xtrain.append(frame)   # pick up the section in time_window size
        Ytrain.append(data[i+time_lag:(i+time_lag + 1)].T)       # pick up the next one as the prediction
    
    Xtrain = np.array(Xtrain).reshape(len(Xtrain),time_lag+1)    # Convert them from list to array   
    Ytrain = np.array(Ytrain).reshape(len(Ytrain),1)
    
    
    model = Sequential() 
    # Adding layers to neural net
    model.add(Dense(neurons, input_dim=time_lag+1, activation='relu'))
    model.add(Dense(50, input_dim=neurons, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam') # Set optimisation method
    
    # Training
    model.fit(Xtrain, Ytrain, epochs=100, batch_size=2, verbose=2, validation_split=0.05)
    
    
    # Predicting
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
    return testPredict

#%% Combination forecast
def comb_fore(df):
    # log data for Arima
    lg_data = np.log(df)
    # calculate residuals for arima
    sarima_model1 = sm.tsa.statespace.SARIMAX(lg_data, order=(2,1,1), seasonal_order=(2,1,1,12))
    result1 = sarima_model1.fit(disp=False)
    residual1 = df - np.exp(result1.fittedvalues)
    residual1[12] = residual1[24]
    
    # calculate residuals for multiplicative decomp
    res = sm.tsa.seasonal_decompose(df,model='multiplicative')
    x = np.array(range(6,len(df)-6))
    y = np.array(res.trend)
    y = y[~np.isnan(y)]
    trend_fit = np.poly1d(np.polyfit(x, y, deg = 1))
    residual2 = df.values[6:-6] - trend_fit(x)*res.seasonal[6:-6]
    
    # Calculate Variance Coveriance
    covariance = np.cov(residual1[6:-6], residual2)
    var1 = covariance[0][0]
    var2 = covariance[1][1] 
    rho = covariance[0][1] / (np.sqrt(var1*var2))
    
    # Optimise weights based on variance
    wopt1 = (var2 - rho*np.sqrt(var1*var2))/(var1+var2-2*rho*np.sqrt(var1*var2))
    wopt2 = 1 - wopt1;
    
    # Forecast using arima
    forecasts1 = result1.forecast(6)
    forecasts1 = np.exp(forecasts1)
    
    # Forecast using multiplicative decomp
    pred_val = np.array(range(len(df), len(df)+6))
    pred_trend = trend_fit(pred_val)
    forecasts2 = np.multiply(pred_trend,res.seasonal[pred_val - 12])
 
    # Calculate combined forecast
    forecasts = wopt1*forecasts1 + wopt2*forecasts2.values
    return forecasts

#%% Read Data
file = 'ele_card_total.csv'

df = pd.read_csv(file)


df['Date'] = [df['Date'][i].replace('M','/') for i in range(len(df['Date']))]

df['Date'] = [dt.datetime.strptime(date, '%Y/%m').date() for date in df['Date']]

dates = list(df['Date'])
values = list(df['Total'])
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
data = df.Total

#%% Cross Validation

# Declare error metricx
nh = 6
k = 113
n = len(data)
mape1 = np.empty((10,6))
mape2 = np.empty((10,6))
mape3 = np.empty((10,6))
mape4 = np.empty((10,6))
mape5 = np.empty((10,6))

mae1 = np.empty((10,6))
mae2 = np.empty((10,6))
mae3 = np.empty((10,6))
mae4 = np.empty((10,6))
mae5 = np.empty((10,6))

mse1 = np.empty((10,6))
mse2 = np.empty((10,6))
mse3 = np.empty((10,6))
mse4 = np.empty((10,6))
mse5 = np.empty((10,6))

for i in range(10):
    xtrain = data[:k+nh*i+1]
    ytrain = data[k+nh*i+1:k+nh*i+1+nh]
    pred = baseline(xtrain)
    mae1[i,:] = np.absolute(pred - ytrain.values)
    mape1[i,:] = np.absolute(pred - ytrain.values)/ytrain.values
    mse1[i,:] = (pred - ytrain.values)**2
    pred = Multiplicative(xtrain)
    mae2[i,:] = np.absolute(pred - ytrain.values)
    mape2[i,:] = np.absolute(pred - ytrain.values)/ytrain.values
    mse2[i,:] = (pred - ytrain.values)**2
    pred = sarima(xtrain)
    mae3[i,:] = np.absolute(pred - ytrain.values)
    mape3[i,:] = np.absolute(pred - ytrain.values)/ytrain.values
    mse3[i,:] = (pred - ytrain.values)**2
    pred = NN(xtrain, 12, 100)
    mae4[i,:] = np.absolute(pred - ytrain.values)
    mape4[i,:] = np.absolute(pred - ytrain.values)/ytrain.values
    mse4[i,:] = (pred - ytrain.values)**2
    pred = comb_fore(xtrain)
    mae5[i,:] = np.absolute(pred - ytrain.values)
    mape5[i,:] = np.absolute(pred - ytrain.values)/ytrain.values
    mse5[i,:] = (pred - ytrain.values)**2
