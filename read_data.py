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
from holtwinters import linear          # For simple exponential smoothing
from holtwinters import multiplicative  # For multiplcative decomposition

def perdelta(start, end, delta):
    curr = start
    while curr < end:
        yield curr
        curr += delta

#%% Read data

#file = 'food_price_index.csv'
file = 'ele_card_total.csv'

df = pd.read_csv(file)


df['Date'] = [df['Date'][i].replace('M','/') for i in range(len(df['Date']))]

df['Date'] = [dt.datetime.strptime(date, '%Y/%m').date() for date in df['Date']]

#
#for columns in df:
#    plt.figure()
#    plt.plot(df['Date'],df[columns])
#    plt.title(columns)
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

pred_val = np.array(range(len(df), len(df)+8))

pred_trend = trend_fit(pred_val)

pred = np.multiply(pred_trend,seasonal_index[3:11])

date = []

end = dt.date(2017,11,1)
current = dt.date(2017, 4, 1)    

# Generate dates
while current<= end:
    date.append(current)
    current += relativedelta(months=1)

plt.figure()
plt.plot(dates, values, label = 'Observed')
plt.plot(date,pred, label = 'Predicted')
plt.ylabel('Amount in Millions')
plt.xlabel('Time')
plt.legend(loc='upper left')
plt.title('Multiplicative decomposition Forecasts')

#%% Holt Winters

x_smoothed, Y, alpha, beta, rmse = linear(values, 8)
plt.figure()
plt.plot(dates, values, label = 'Observed')
plt.plot(date, Y, label = 'Predicted')
plt.ylabel('Amount in Millions')
plt.xlabel('Time')
plt.legend(loc='upper left')
plt.title('Linear Holts Winter Forecasts')


x_smoothed, Y, s, alpha, beta, gamma, rmse = multiplicative(values, 12, 8)
plt.figure()
plt.plot(dates, values, label = 'Observed')
plt.plot(date, Y, label = 'Predicted')
plt.ylabel('Amount in Millions')
plt.xlabel('Time')
plt.legend(loc='upper left')
plt.title('Multiplicative Holts Winter Forecasts')

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

fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(211)
ax1.plot(lg_data_ds)
plt.title('Seasonally Differenced Electronic Payment')
ax2 = fig.add_subplot(212)
ax2.plot(lg_data_dsd)
plt.title('Regular Difference of the Seasonal Differenced')

## ACF & PACF plots
#fig = plt.figure(figsize=(12,9))
#ax1 = fig.add_subplot(211)
#fig = sm.graphics.tsa.plot_acf(lg_data, lags=40, ax=ax1)
#ax1.set_title("ACF: Log Data")
#ax2 = fig.add_subplot(212)
#fig = sm.graphics.tsa.plot_pacf(lg_data, lags=40, ax=ax2)
#ax2.set_title("PACF: Log Data")
#
#fig = plt.figure(figsize=(12,9))
#ax1 = fig.add_subplot(211)
#fig = sm.graphics.tsa.plot_acf(lg_data_d, lags=40, ax=ax1)
#ax1.set_title("ACF: first order difference of log data")
#ax2 = fig.add_subplot(212)
#fig = sm.graphics.tsa.plot_pacf(lg_data_d, lags=40, ax=ax2)
#ax2.set_title("PACF: first order difference of log data")
#
#fig = plt.figure(figsize=(12,9))
#ax1 = fig.add_subplot(211)
#fig = sm.graphics.tsa.plot_acf(lg_data_ds, lags=40, ax=ax1)
#ax1.set_title("ACF: seasonal difference of log data")
#ax2 = fig.add_subplot(212)
#fig = sm.graphics.tsa.plot_pacf(lg_data_ds, lags=40, ax=ax2)  
#ax2.set_title("PACF: seasonal difference of log data")   
#
#fig = plt.figure(figsize=(12,9))
#ax1 = fig.add_subplot(211)
#fig = sm.graphics.tsa.plot_acf(lg_data_dsd, lags=40, ax=ax1)
#ax1.set_title("ACF: seasonal difference of log data")
#ax2 = fig.add_subplot(212)
#fig = sm.graphics.tsa.plot_pacf(lg_data_dsd, lags=40, ax=ax2)  
#ax2.set_title("PACF: seasonal difference of log data") 


sarima_model = sm.tsa.statespace.SARIMAX(lg_data, order=(1,0,3), seasonal_order=(0,1,1,12))  

result = sarima_model.fit(disp=False)
print(result.summary())

# Forecasting
forecasts = result.forecast(8)

# Display forecasting
plt.figure()
plt.plot(dates[6:], lg_data.values, label = 'Observed')
plt.plot(date,forecasts, label = 'Predicted')
plt.ylabel('Amount in Millions')
plt.xlabel('Time')
plt.legend(loc='upper left')
plt.title('Multiplicative decomposition Forecasts')