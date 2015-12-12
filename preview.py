from __future__ import print_function # python3 print function

__author__   = 'Joseph Cheng <indiejoseph@gmail.com>'
__version__  = '0.1'

import os
from data_prepare import YahooHistorical
from datetime import date, datetime
from sys import stdout
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

def normalize(data, mean=None, std=None):
  if mean == None:
    mean = np.mean(data)
  if std == None:
    std = np.std(data)
  x = (data - mean) / std
  return x, mean, std

# prepare date
symbol = '^HSI'
yahoo_data = YahooHistorical(data_from=date(2014, 1, 1), data_to=date(2015, 12, 31))
yahoo_data.open(os.path.join(os.path.dirname(__file__), 'data/' + symbol + '.csv'))
test_set = yahoo_data.get()
rsi = yahoo_data.relative_strength(n=14)
(sma13, sma7, macd) = yahoo_data.moving_average_convergence(7, 13) # 7 days and 13 days moving average
label = np.array([n['date'] for n in test_set])
(prices, m, s) = normalize(np.array([n['adj_close'] for n in test_set]))
(tmax, tmin) = yahoo_data.trading_range_breakout(50)
(sma13, m, s) = normalize(sma13, m, s)
(sma7, m, s) = normalize(sma7, m, s)
(tmax, m, s) = normalize(tmax, m, s)
(tmin, m, s) = normalize(tmin, m, s)
(macd, m, s) = normalize(macd)
(rsi, m, s) = normalize(rsi)
emax = list(argrelextrema(prices, np.greater)[0])
emin = list(argrelextrema(prices, np.less)[0])

plt.grid(True)
plt.plot(label, prices, color="blue", label=symbol)
plt.fill_between(label, prices, np.min(prices), color="blue", alpha=.15)
plt.plot(label, sma7[-len(prices):], color="green", label='SMA(7)')
plt.plot(label, sma13[-len(prices):], color="red", label='SMA(13)')
plt.plot(label, macd[-len(prices):], color="cyan", linestyle="--", label='MACD')
plt.plot(label, rsi[-len(prices):], color="orange", label='RSI')
plt.plot(label, tmax[-len(prices):], color="green", linestyle="--", label='TRB Max')
plt.plot(label, tmin[-len(prices):], color="red", linestyle="--", label='TRB Min')
plt.scatter(label[emax], prices[emax], color="b", marker="o", label="Sell")
plt.scatter(label[emin], prices[emin], color="r", marker="*", label="Buy")

plt.legend()

plt.show()
