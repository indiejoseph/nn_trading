from __future__ import print_function # python3 print function

__author__   = 'Joseph Cheng <indiejoseph@gmail.com>'
__version__  = '0.1'

import os
from data_prepare import YahooHistorical
import numpy as np
from scipy import random
import talib
from datetime import date, datetime
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

def normalize(data, mean=None, std=None):
  if mean == None:
    mean = np.mean(data)
  if std == None:
    std = np.std(data)
  x = (data - mean) / std
  return x, mean, std

def ir(p, i, data): # p-days, i=current day
  a = np.sum([n['adj_close'] for n in data[i:i+p]]) / p
  c = data[i]['adj_close']
  return ((a - c) / c)

# prepare date
symbol = '0700.HK'
yahoo_data = YahooHistorical(data_from=date(2014, 1, 1), data_to=date(2015, 12, 31))
yahoo_data.open(os.path.join(os.path.dirname(__file__), 'data/' + symbol + '.csv'))
test_set = yahoo_data.get()
rsi = yahoo_data.relative_strength(n=14)
(sma13, sma7, macd) = yahoo_data.moving_average_convergence(7, 13) # 7 days and 13 days moving average and MACD
test_label = np.array([n['date'] for n in test_set])
(prices, tmean, tstd) = normalize(np.array([n['adj_close'] for n in test_set]))
test_target = np.zeros(len(prices))
emax = list(argrelextrema(prices, np.greater)[0])
emin = list(argrelextrema(prices, np.less)[0])
(tmax50, tmin50) = yahoo_data.trading_range_breakout(50)
(tmax100, tmin100) = yahoo_data.trading_range_breakout(100)
(tmax200, tmin200) = yahoo_data.trading_range_breakout(200)
(tmax50, a, s) = normalize(tmax50, tmean, tstd)
(tmin50, a, s) = normalize(tmin50, tmean, tstd)
(tmax100, a, s) = normalize(tmax100, tmean, tstd)
(tmin100, a, s) = normalize(tmin100, tmean, tstd)
(tmax200, a, s) = normalize(tmax200, tmean, tstd)
(tmin200, a, s) = normalize(tmin200, tmean, tstd)
(sma7, s20mean, s20std) = normalize(sma7, tmean, tstd)
(sma13, s100mean, s100std) = normalize(sma13, tmean, tstd)
(macd, macdmean, macdstd) = normalize(macd)

print(len(emax))
print(len(emin))

(rsi, rsimean, rsistd) = normalize(rsi)
plt.grid(True)
plt.plot(test_label, prices, color="blue", label=symbol)
plt.fill_between(test_label, prices, np.min(prices), color="blue", alpha=.15)
plt.plot(test_label, sma7[-len(prices):], color=random.rand(3,1), label='SMA(7)')
plt.plot(test_label, sma13[-len(prices):], color=random.rand(3,1), label='SMA(13)')
plt.plot(test_label, macd[-len(prices):], color=random.rand(3,1), linestyle="--", label='MACD')
plt.plot(test_label, rsi[-len(prices):], color=random.rand(3,1), label='RSI')
plt.plot(test_label, tmax50[-len(prices):], color=random.rand(3,1), label='TRB50 Max')
plt.plot(test_label, tmin50[-len(prices):], color=random.rand(3,1), label='TRB50 Min')
plt.plot(test_label, tmax100[-len(prices):], color=random.rand(3,1), label='TRB100 Max')
plt.plot(test_label, tmin100[-len(prices):], color=random.rand(3,1), label='TRB100 Min')
plt.plot(test_label, tmax200[-len(prices):], color=random.rand(3,1), label='TRB200 Max')
plt.plot(test_label, tmin200[-len(prices):], color=random.rand(3,1), label='TRB200 Min')
plt.scatter(test_label[emax], prices[emax], color="b", marker="o", label="Sell")
plt.scatter(test_label[emin], prices[emin], color="r", marker="*", label="Buy")


plt.legend()

plt.show()
