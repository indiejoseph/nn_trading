from __future__ import print_function # python3 print function

__author__ = 'Joseph Cheng <indiejoseph@gmail.com>'
__version__ = '0.1'

import os
import sys
from data_prepare import YahooHistorical
import matplotlib.pyplot as plt
import numpy as np
import random
import talib
from scipy.signal import argrelextrema
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM

np.random.seed(12)
random.seed(12)

def preprocessing(p, data):
  x = []
  y = []
  for i in xrange(p, len(data)-1):
    x.append(data[i-p:i])
    tdy = data[i-1][0]
    tmr = data[i][0]
    target = (tmr - tdy) / tdy
    y.append([target])
  x = np.asarray(x)
  y = np.asarray(y)
  return x, y

def normalize(data, mean=None, std=None):
  # Decimal Scaling
  # replace nan with first non-nan
  nan = np.isnan(data)
  nnan = np.where(~nan)[0][0]
  data[:nnan] = data[nnan]

  if mean == None:
    mean = np.mean(data)
  if std == None:
    std = np.std(data)
  return (data - mean) / std, mean, std

if __name__=="__main__":
  # prepare date
  symbol = "0005.HK"
  stock_data = YahooHistorical()
  stock_data.open(os.path.join(os.path.dirname(__file__), "data/" + symbol + ".csv"))
  dataset = stock_data.get()
  date = np.array([n['date'] for n in dataset])
  close_prices = np.array([n['adj_close'] for n in dataset])
  low_prices = np.array([n['low'] for n in dataset])
  high_prices = np.array([n['high'] for n in dataset])
  volumes = np.array([n['vol'] for n in dataset])
  prices, mean, std = normalize(close_prices)
  low_prices, mean, std = normalize(low_prices, mean, std)
  high_prices, mean, std = normalize(high_prices, mean, std)
  emax = list(argrelextrema(prices, np.greater)[0])
  emin = list(argrelextrema(prices, np.less)[0])
  sma13, mean, std = normalize(talib.SMA(close_prices, 13), mean, std) # 50-day SMA
  sma50, mean, std = normalize(talib.SMA(close_prices, 50), mean, std) # 50-day SMA
  sma100, mean, std = normalize(talib.SMA(close_prices, 100), mean, std) # 100-day SMA
  sma200, mean, std = normalize(talib.SMA(close_prices, 200), mean, std) # 200-day SMA
  macd_upper, macd_middle, macd_lower = talib.MACD(close_prices, 12, 26, 9)
  macd_middle, macd_mean, macd_std = normalize(macd_middle)
  macd_upper, macd_mean, macd_std = normalize(macd_upper, macd_mean, macd_std)
  macd_lower, macd_mean, macd_std = normalize(macd_lower, macd_mean, macd_std)
  mfi, mfi_mean, mfi_std  = normalize(talib.MFI(high_prices, low_prices, close_prices, volumes)) # Money Flow Index
  rsi, rsi_mean, rsi_std  = normalize(talib.RSI(close_prices))
  volumes, vol_mean, vol_std = normalize(volumes)
  skip = 200
  data = zip(prices[skip:],\
            low_prices[skip:],\
            high_prices[skip:],\
            sma50[skip:],\
            sma100[skip:],\
            sma200[skip:],\
            macd_upper[skip:],\
            macd_middle[skip:],\
            macd_lower[skip:],\
            mfi[skip:],\
            rsi[skip:])
  p = 17
  batch_size = 100
  epochs = 1000
  training_x, training_y = preprocessing(p, data)
  input_size = len(training_x[0][0])
  lstm_size = 32

  # build model
  model = Sequential()
  model.add(LSTM(lstm_size, return_sequences=True, input_shape=(p, input_size)))
  model.add(Activation('tanh'))
  model.add(Dropout(0.2))
  model.add(LSTM(lstm_size, return_sequences=True))
  model.add(Activation('tanh'))
  model.add(Dropout(0.2))
  model.add(LSTM(lstm_size, return_sequences=False))
  model.add(Activation('tanh'))
  model.add(Dropout(0.2))
  model.add(Dense(len(training_y[0])))
  model.compile(loss="mean_squared_error", optimizer="adam")
  model.load_weights("train_lstm.nn")

  # train
  print('training...')
  # model.fit(training_x, training_y, batch_size=batch_size, nb_epoch=epochs, verbose=1, validation_split=0.2)
  # model.save_weights(os.path.join("train_lstm.nn"), overwrite=True)

  # test
  test_data = data[int(len(data)*(1-0.2)):]
  test_x, test_y = preprocessing(p, test_data)
  label = range(len(test_x))
  pred = model.predict(test_x, verbose=1, batch_size=batch_size)
  test_price = []
  predict = []
  for i in xrange(len(test_x)):
    x = test_x[i][-1][0]
    test_price.append(x)
    predict.append(x + (x * pred[i][0]))

  # print
  plt.grid(True)
  plt.plot(label, test_price, color=np.random.rand(3,1), label="prices")
  plt.plot(label, predict, color=np.random.rand(3,1), label="predict")
  # plt.plot(date[skip:], prices[skip:], color=np.random.rand(3,1), label="prices")
  # plt.plot(date[skip:], low_prices[skip:], color=np.random.rand(3,1), label="low_prices")
  # plt.plot(date[skip:], high_prices[skip:], color=np.random.rand(3,1), label="high_prices")
  # plt.plot(date[skip:], sma50[skip:], color=np.random.rand(3,1), linestyle="--", label="sma50")
  # plt.plot(date[skip:], sma100[skip:], color=np.random.rand(3,1), linestyle="--", label="sma100")
  # plt.plot(date[skip:], sma200[skip:], color=np.random.rand(3,1), linestyle="--", label="sma200")
  # plt.plot(date[skip:], macd_upper[skip:], color=np.random.rand(3,1), linestyle="--", label="macd_upper")
  # plt.plot(date[skip:], macd_middle[skip:], color=np.random.rand(3,1), linestyle="--", label="macd_middle")
  # plt.plot(date[skip:], macd_lower[skip:], color=np.random.rand(3,1), linestyle="--", label="macd_lower")
  # plt.plot(date[skip:], mfi[skip:], color=np.random.rand(3,1), linestyle="--", label="mfi")
  # plt.plot(date[skip:], rsi[skip:], color=np.random.rand(3,1), linestyle="--", label="rsi")
  plt.legend()
  plt.show()
