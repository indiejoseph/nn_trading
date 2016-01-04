import talib
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt

class IndexBase(object):
  def __init__(self, prices):
    self.prices = prices
    self.n = len(prices)
    self.result = 0
    self.mean = None
    self.std = None
    self.computeIndex()

  def computeIndex(self):
    pass

  def getResult(self):
    return self.result

  def normalize(self):
    values = self.result
    output = []

    for i in xrange(len(values)):
      value = values[i]

      # fill all nans with first non-nan
      non_nan = filter(lambda x: np.isnan(x) == False, value)[0]
      value[np.isnan(value)] = non_nan

      if self.mean == None:
        self.mean = np.mean(value)
      if self.std == None:
        self.std = np.std(value)

      # Z-score
      output.append((value - self.mean) / self.std)

    return output


class ClosePriceIndexBase(IndexBase):
  def __init__(self, prices):
    self.prices = prices
    self.n = len(prices)
    self.result = 0
    self.mean = None
    self.std = None
    self.computeIndex()


class ClosePriceIndexWithSharedMeanBase(IndexBase):
  def __init__(self, prices, mean=None, std=None):
    self.prices = prices
    self.n = len(prices)
    self.result = 0
    self.mean = mean
    self.std = std
    self.computeIndex()


class ClosePriceIndexWithPeriodSharedMeanBase(IndexBase):
  def __init__(self, prices, period, mean=None, std=None):
    self.prices = prices
    self.n = len(prices)
    self.period = period
    self.result = 0
    self.mean = mean
    self.std = std
    self.computeIndex()


class ClosePriceIndexWithFastSlowSignalPeriodBase(IndexBase):
  def __init__(self, prices, fast_period=None, slow_period=None, signal_period=None):
    self.prices = prices
    self.n = len(prices)
    self.fast_period = fast_period
    self.slow_period = slow_period
    self.signal_period = signal_period
    self.result = 0
    self.mean = None
    self.std = None
    self.computeIndex()


class AllPricesIndexBase(IndexBase):
  def __init__(self, low_prices, high_prices, close_prices):
    self.low_prices = low_prices
    self.high_prices = high_prices
    self.close_prices = close_prices
    self.n = len(low_prices)
    self.result = 0
    self.mean = None
    self.std = None
    self.computeIndex()


class AllPricesAndVolumeIndexBase(IndexBase):
  def __init__(self, low_prices, high_prices, close_prices, volumes):
    self.low_prices = low_prices
    self.high_prices = high_prices
    self.close_prices = close_prices
    self.volumes = volumes
    self.n = len(low_prices)
    self.result = 0
    self.mean = None
    self.std = None
    self.computeIndex()


class SMAIndex(ClosePriceIndexWithPeriodSharedMeanBase):
  def computeIndex(self):
    output = talib.SMA(self.prices, self.period)
    self.result = (output,)
    return self.result


class EMAIndex(ClosePriceIndexWithPeriodSharedMeanBase):
  def computeIndex(self):
    output = talib.EMA(self.prices, self.period)
    self.result = (output,)
    return self.result


class MACDIndex(ClosePriceIndexWithFastSlowSignalPeriodBase):
  def computeIndex(self):
    upper, middle, lower = talib.MACD(self.prices, self.fast_period, self.slow_period, self.signal_period)
    self.result = (upper, middle, lower)
    return self.result


class RSIIndex(ClosePriceIndexBase):
  def computeIndex(self):
    output = talib.RSI(self.prices)
    self.result = (output,)
    return self.result


class STOCHRSIIndex(ClosePriceIndexBase):
  def computeIndex(self):
    upper, lower = talib.STOCHRSI(self.prices)
    self.result = (upper, lower)
    return self.result


class BBANDSIndex(ClosePriceIndexBase):
  def computeIndex(self):
    upper, middle, lower = talib.BBANDS(self.prices, matype=talib.MA_Type.T3)
    self.result = (upper, middle, lower)
    return self.result


class ADXIndex(AllPricesIndexBase):
  def computeIndex(self):
    output = talib.ADX(self.high_prices, self.low_prices, self.close_prices)
    self.result = (output,)
    return self.result


class STOCHIndex(AllPricesIndexBase):
  def computeIndex(self):
    upper, lower = talib.STOCH(self.high_prices, self.low_prices, self.close_prices)
    self.result = (upper, lower)
    return self.result


class MFIIndex(AllPricesAndVolumeIndexBase):
  def computeIndex(self):
    output = talib.MFI(self.high_prices, self.low_prices, self.close_prices, self.volumes)
    self.result = (output,)
    return self.result


class ADOSCIndex(AllPricesAndVolumeIndexBase):
  def computeIndex(self):
    output = talib.ADOSC(self.high_prices, self.low_prices, self.close_prices, self.volumes)
    self.result = (output,)
    return self.result


class Features(object):
  def __init__(self, open_prices, low_prices, high_prices, close_prices, volumes):
    self.open_prices = open_prices
    self.low_prices = low_prices
    self.high_prices = high_prices
    self.close_prices = close_prices
    self.volumes = volumes
    self.mean = None
    self.std = None
    self.num_output = 2 # high or low trend

    # indices
    self.indices = [
      SMAIndex(close_prices, 10, self.mean, self.std), # 10-day SMA
      SMAIndex(close_prices, 50, self.mean, self.std), # 50-day SMA
      SMAIndex(close_prices, 200, self.mean, self.std), # 200-day SMA
      MACDIndex(close_prices, 26, 12, 9), # MCDA
      RSIIndex(close_prices), # RSI
      STOCHRSIIndex(close_prices), # Stoch RSI
      BBANDSIndex(close_prices), # Bollinger Bands
      ADXIndex(high_prices, low_prices, close_prices), # ADX
      STOCHIndex(high_prices, low_prices, close_prices),
      MFIIndex(high_prices, low_prices, close_prices, volumes), # Money Flow Index
      # ADOSCIndex(high_prices, low_prices, close_prices, volumes) # ADOSC
    ]

  def normalize(self, data):
    data = np.asarray(data)

    if self.mean == None:
      self.mean = np.mean(data)

    if self.std == None:
      self.std = np.std(data)

    return (data - self.mean) / self.std

  def getInput(self):
    data = [
      (self.normalize(self.open_prices),),
      (self.normalize(self.high_prices),),
      (self.normalize(self.low_prices),),
      (self.normalize(self.close_prices),)
    ]

    for index in self.indices:
      data.append(index.normalize())

    data = zip(*list(chain.from_iterable(data)))

    return data

  def getOutput(self):
    data = []
    num = len(self.close_prices)

    for i, price in enumerate(self.close_prices):
      # short
      if i < num-1:
        trend = (self.close_prices[i+1] - price) / price
        if trend >= 0.01:
          trend = 0 # UP
        elif trend <= -0.01:
          trend = 1 # DOWN
        else:
          trend = 2 # NOWHERE
        data.append(trend)

    # fill last n price
    data.append(data[-1])

    return data


if __name__ == '__main__':
  n = 500
  x = range(n)
  open_prices = np.random.random(n)
  close_prices = np.random.random(n)
  high_prices = np.random.random(n)
  low_prices = np.random.random(n)
  volumes = np.random.random(n)
  features = Features(open_prices, high_prices, low_prices, close_prices, volumes)
  y = features.normalize(close_prices)

  plt.grid(True)
  plt.plot(x, y, color="blue", label="Actual")
  plt.fill_between(x, y, np.min(y), color="blue", alpha=.15)
  plt.plot(x, features.getInput(), color="olive", label="Input")
  plt.plot(x, features.getOutput(), color="pink", label="Output")
  plt.legend()
  plt.show()
