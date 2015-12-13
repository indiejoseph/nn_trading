from __future__ import print_function # python3 print function

__author__ = 'Joseph Cheng <indiejoseph@gmail.com>'
__version__ = '0.1'

from data_prepare import YahooHistorical

import os
from datetime import date
import matplotlib.pyplot as plt
from scipy import random
import numpy as np
import talib
import math
import functools

class IndicesContainer:
  def __init__(self, low_prices, high_prices, close_prices, volumes):
    self.indicesList = [
      SMASortIndex(close_prices),
      SMALongIndex(close_prices),
      EMAIndex(close_prices),
      MACDIndex(close_prices),
      RSIIndex(close_prices),
      STOCHRSIIndex(close_prices),
      BBANDSIndex(close_prices),
      ADXIndex(low_prices, high_prices, close_prices),
      STOCHIndex(low_prices, high_prices, close_prices),
      ADOSCIndex(low_prices, high_prices, close_prices, volumes),
      MFIIndex(low_prices, high_prices, close_prices, volumes)
    ]

  def computeIndices(self):
    for index in self.indicesList:
      index.computeIndex()

  def getIndicesList(self):
    return self.indicesList


class ClosePriceIndexBase:
  def __init__(self, prices):
    self.prices = prices
    self.n = len(prices)
    self.result = 0

  def computeIndex(self):
    pass

  def getResult(self):
    return self.result


class AllPricesIndexBase:
  def __init__(self, low_prices, high_prices, close_prices):
    self.low_prices = low_prices
    self.high_prices = high_prices
    self.close_prices = close_prices
    self.n = len(low_prices)
    self.result = 0

  def computeIndex(self):
    pass

  def getResult(self):
    return self.result


class AllPricesAndVolumeIndexBase():
  def __init__(self, low_prices, high_prices, close_prices, volumes):
    self.low_prices = low_prices
    self.high_prices = high_prices
    self.close_prices = close_prices
    self.volumes = volumes
    self.n = len(low_prices)
    self.result = 0

  def computeIndex(self):
    pass

  def getResult(self):
    return self.result


class SMAIndex(ClosePriceIndexBase):
  def computeIndex(self):
    output = talib.SMA(self.prices)
    self.result = output
    return self.result


class EMAIndex(ClosePriceIndexBase):
  def computeIndex(self):
    output = talib.EMA(self.prices)
    self.result = output
    return self.result


class MACDIndex(ClosePriceIndexBase):
  def computeIndex(self):
    upper, middle, lower = talib.MACD(self.prices)
    self.result = middle
    return self.result


class RSIIndex(ClosePriceIndexBase):
  def computeIndex(self):
    output = talib.RSI(self.prices)
    self.result = output
    return self.result


class STOCHRSIIndex(ClosePriceIndexBase):
  def computeIndex(self):
    upper, lower = talib.STOCHRSI(self.prices)
    self.result = upper
    return self.result


class BBANDSIndex(ClosePriceIndexBase):
  def computeIndex(self):
    upper, middle, lower = talib.BBANDS(self.prices, matype=talib.MA_Type.T3)
    self.result = middle
    return self.result


class ADXIndex(AllPricesIndexBase):
  def computeIndex(self):
    output = talib.ADX(self.high_prices, self.low_prices, self.close_prices)
    self.result = output
    return self.result


class STOCHIndex(AllPricesIndexBase):
  def computeIndex(self):
    upper, lower = talib.STOCH(self.high_prices, self.low_prices, self.close_prices)
    self.result = upper
    return self.result


class MFIIndex(AllPricesAndVolumeIndexBase):
  def computeIndex(self):
    output = talib.MFI(self.high_prices, self.low_prices, self.close_prices, self.volumes)
    self.result = output
    return self.result


class ADOSCIndex(AllPricesAndVolumeIndexBase):
  def computeIndex(self):
    output = talib.ADOSC(self.high_prices, self.low_prices, self.close_prices, self.volumes)
    self.result = output
    return self.result


class indicesNormalizer(object):
  def normalize(self, indices_values):
    highest_non_nan = 0
    for indexlist in indices_values:
      for element in range(len(indexlist)):
        if math.isnan(indexlist[element]):
          if element > highest_non_nan:
            highest_non_nan = element
        else:
          break
          #print "Highest non nan:", highest_non_nan

    normalized_list = []
    for indexlist in indices_values:
      normalized_list.append(indexlist[highest_non_nan + 1:])

    return normalized_list


def pickIndices(pickFrom, templateIndicesSet):
  result = []

  for index in pickFrom:
    for template_index in templateIndicesSet:
      if index.__class__.__name__ == template_index.__class__.__name__:
        result.append(index)
        break

  return result


class TrendFinderException(Exception):
  def __init__(self, value):
    self.value = value

  def __str__(self):
    return repr(self.value)


def findSellTrendBeginning(close_prices, jump_limit=float('inf')):
  assert(len(close_prices))
  prev_price = close_prices[0]
  max_jump = 0
  max_jump_ind = 0

  for i in range(len(close_prices)):
    if close_prices[i] - prev_price > max_jump and math.fabs(close_prices[i] - prev_price < jump_limit):
      max_jump = close_prices[i] - prev_price
      max_jump_ind = i
    prev_price = close_prices[i]

  if max_jump_ind - 1 < 0:
    raise TrendFinderException("Couldn't find desired trend!")

  return max_jump_ind - 1 # because the indices, that can be used to determine the jump, can be computed from the previous price


def findBuyTrendBeginning(close_prices, jump_limit=float('inf')):
  prev_price = close_prices[0]
  max_jump = 0
  max_jump_ind = 0

  for i in range(len(close_prices)):
    if close_prices[i] - prev_price < max_jump and math.fabs(close_prices[i] - prev_price) < jump_limit:
      max_jump = close_prices[i] - prev_price
      max_jump_ind = i
    prev_price = close_prices[i]

  if max_jump_ind - 1 < 0:
    raise TrendFinderException("Couldn't find desired trend!")

  return max_jump_ind - 1 # because the indices, that can be used to determine the jump, can be computed from the previous price


class nonNanTrendFinder(object):
  def utilize(self, trendFinderfunctor, indices_list):
    retry_counter = 1
    trendBeginning = trendFinderfunctor()

    isNonNanVector = True
    for index in indices_list:
      if math.isnan(index.getResult()[trendBeginning]):
        isNonNanVector = False
        break

    while isNonNanVector == False:
      if retry_counter == 10:
        raise TrendFinderException("Couldn't find desired trend!")
      else:
        trendBeginning = trendFinderfunctor.func(trendFinderfunctor.args[0],
          math.fabs(
            trendFinderfunctor.args[0][trendBeginning + 1] - trendFinderfunctor.args[0][trendBeginning]))
        isNonNanVector = True
        for index in indices_list:
          if math.isnan(index.getResult()[trendBeginning]):
            isNonNanVector = False
            break
      retry_counter += 1

    return trendBeginning


class simpleTrendBeginningsFinder(object):
  def __init__(self, close_prices, indices_list):
    assert(len(close_prices))
    self.close_prices = close_prices
    self.indices_list = indices_list

  def findTrendBeginnings(self):
    sellTrendFunctor = functools.partial(findSellTrendBeginning, self.close_prices)
    buyTrendFunctor = functools.partial(findBuyTrendBeginning, self.close_prices)
    nonNanSellTrendBeginning = nonNanTrendFinder().utilize(sellTrendFunctor, self.indices_list)
    nonNanBuyTrendBeginning = nonNanTrendFinder().utilize(buyTrendFunctor, self.indices_list)

    return (nonNanSellTrendBeginning, nonNanBuyTrendBeginning)

if __name__ == '__main__':
  symbol = '^HSI'
  yahoo_data = YahooHistorical(data_from=date(2014, 1, 1), data_to=date(2015, 12, 31))
  yahoo_data.open(os.path.join(os.path.dirname(__file__), 'data/' + symbol + '.csv'))
  training_data = yahoo_data.get()
  training_low_prices = np.asarray([n['low'] for n in training_data])
  training_high_prices = np.asarray([n['high'] for n in training_data])
  training_close_prices = np.asarray([n['close'] for n in training_data])
  training_open_prices = np.asarray([n['open'] for n in training_data])
  training_volumes = np.asarray([n['vol'] for n in training_data])
  training_label = [n['date'] for n in training_data]
  indicesContainer = IndicesContainer(training_low_prices, training_high_prices,
    training_close_prices, training_volumes)
  indicesContainer.computeIndices()
  indicesList = indicesContainer.getIndicesList()

  plt.plot(training_label, training_close_prices, color="blue", label=symbol)
  plt.fill_between(training_label, training_close_prices, np.min(training_close_prices), color="blue", alpha=.15)

  for index in indicesList:
    print(index.getResult())
    plt.plot(training_label, index.getResult(), color=random.rand(3,1))

  plt.legend()
  plt.show()
