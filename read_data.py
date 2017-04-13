#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 18:22:40 2017

@author: Max
"""

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

#file = 'food_price_index.csv'
file = 'ele_card.csv'

df = pd.read_csv(file)

df['Date'] = [df['Date'][i].replace('M','/') for i in range(len(df['Date']))]

df['Date'] = [dt.datetime.strptime(date, '%Y/%m').date() for date in df['Date']]

for columns in df:
    plt.figure()
    plt.plot(df['Date'],df[columns])
    plt.title(columns)


