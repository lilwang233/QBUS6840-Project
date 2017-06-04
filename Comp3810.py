#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:20:02 2017

@author: Max
"""

import pandas as pd
import datetime as dt
import statsmodels.api as sm
import numpy as np
from holtwinters import multiplicative  # For multiplcative decomposition
import warnings


def Multiplicative(df):
    x_smoothed, Y, s, alpha, beta, gamma, rmse = multiplicative(list(df.values), 12, 6)
    return Y

def sarima(df):
    
    # Log data
    lg_data = df.copy()
    lg_data = np.log(df)
    
    warnings.filterwarnings("ignore")
    
    sarima_model = sm.tsa.statespace.SARIMAX(lg_data, order=(2,0,0), seasonal_order=(1,1,1,12))  
    
    result = sarima_model.fit(disp=False)
    
    # Forecasting
    forecasts = result.forecast(6)
    forecasts = np.exp(forecasts)
    return forecasts


file = 'ele_card_total.csv'

df = pd.read_csv(file)


df['Date'] = [df['Date'][i].replace('M','/') for i in range(len(df['Date']))]

df['Date'] = [dt.datetime.strptime(date, '%Y/%m').date() for date in df['Date']]


dates = list(df['Date'])
values = list(df['Total'])
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df = df.Total

data = df

nh = 6
k = 113
n = len(data)
mape1 = np.empty((10,6))
mae1 = np.empty((10,6))
mse1 = np.empty((10,6))
mape2 = np.empty((10,6))
mae2 = np.empty((10,6))
mse2 = np.empty((10,6))
mape1[:] = np.nan
mae1[:] = np.nan
mse1[:] = np.nan
mape2[:] = np.nan
mae2[:] = np.nan
mse2[:] = np.nan
for i in range(10):
    xtrain = data[:k+nh*i]
    ytrain = data[k+nh*i+1:k+nh*i+1+nh]
    pred = Multiplicative(xtrain)
    mae1[i,:] = np.absolute(pred - ytrain.values)
    mape1[i,:] = np.absolute(pred - ytrain.values)/ytrain.values
    mse1[i,:] = (pred - ytrain.values)**2
    pred = sarima(xtrain)
    mae2[i,:] = np.absolute(pred - ytrain.values)
    mape2[i,:] = np.absolute(pred - ytrain.values)/ytrain.values
    mse2[i,:] = (pred - ytrain.values)**2