__author__   = 'Joseph Cheng <indiejoseph@gmail.com>'
__version__  = '0.1'

try:
  from urllib.parse import urlencode
except ImportError:
  from urllib import urlencode

import urllib2
import os
import numpy as np
import sys
import codecs
from datetime import date, datetime
import matplotlib.pyplot as plt

# Yahoo Historical
YAHOO_HISTORICAL_URL = 'http://real-chart.finance.yahoo.com/table.csv'

class YahooHistorical:
  def __init__(self, data_from=None, data_to=None):
    self.data_from = data_from
    self.data_to = data_to

  def open(self, file):
    with codecs.open(file, 'r', encoding='utf8') as f:
      csv = f.read()

    return self._parse(csv)


  def download(self, symbol):
    dest_file = symbol + '.csv'
    dest_file = os.path.join(os.path.dirname(__file__), 'data', dest_file)

    query = urlencode({
      's': symbol,
      'g': 'd',
      'ignore': '.csv'
    })

    response = urllib2.urlopen(YAHOO_HISTORICAL_URL + '?' + query).read()
    lines = str(response).split('\n')
    fx = open(dest_file, 'w')

    for line in lines[1:-1]:
      fx.write(line + '\n')
    fx.close()

    return self._parse(response)


  def get(self):
    return self.data

  def trading_range_breakout(self, n):
    prices = [row['adj_close'] for row in self.data]
    tmax = np.zeros(len(prices))
    tmin = np.zeros(len(prices))

    for idx, price in enumerate(prices):
      p = np.append(prices[idx-n:idx+1], price)
      tmax[idx] = np.max(p)
      tmin[idx] = np.min(p)

    return (tmax, tmin)

  def relative_strength(self, n=14):
    """
    compute the n period relative strength indicator
    http://stockcharts.com/school/doku.php?id=chart_school:glossary_r#relativestrengthindex
    http://www.investopedia.com/terms/r/rsi.asp
    """

    prices = [row['adj_close'] for row in self.data]
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed >= 0].sum()/n
    down = -seed[seed < 0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1. + rs)

    for i in range(n, len(prices)):
      delta = deltas[i - 1]  # cause the diff is 1 shorter

      if delta > 0:
        upval = delta
        downval = 0.
      else:
        upval = 0.
        downval = -delta

      up = (up*(n - 1) + upval)/n
      down = (down*(n - 1) + downval)/n

      rs = up/down
      rsi[i] = 100. - 100./(1. + rs)

    return rsi


  def moving_average(self, n, type='simple'):
    """
    compute an n period moving average.

    type is 'simple' | 'exponential'

    """
    prices = [row['adj_close'] for row in self.data]

    if type == 'simple':
      weights = np.ones(n)
    else:
      weights = np.exp(np.linspace(-1., 0., n))

    weights /= weights.sum()

    a = np.convolve(prices, weights, mode='full')[:len(prices)]
    a[:n] = a[n]

    return a

  def moving_average_convergence(self, nslow=26, nfast=12):
    """
    compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    emaslow = self.moving_average(nslow, type='exponential')
    emafast = self.moving_average(nfast, type='exponential')
    return emaslow, emafast, emafast - emaslow

  def _parse(self, data):
    self.data = []
    lines = str(data).split('\n')

    for line in lines[1:-1][::-1]:
      (date, open, high, low, close, vol, adj_close) = line.split(',')

      date = datetime.strptime(date, '%Y-%m-%d').date()
      self.data.append({
        'date': date,
        'open': float(open),
        'high': float(high),
        'low': float(low),
        'close': float(close),
        'vol': float(vol),
        'adj_close': float(adj_close)
      })

    if self.data_from != None:
      self.data = [row for row in self.data if row['date'] >= self.data_from]

    if self.data_to != None:
      self.data = [row for row in self.data if row['date'] <= self.data_to]

    return self.data


if __name__ == '__main__':
  # get historica csv file from last month
  s = '^HSI'
  dest_file = s + '.csv'
  dest_file = os.path.join(os.path.dirname(__file__), 'data', dest_file)
  t = date(2015, 12, 9)
  f = date(2015, 1, 4)
  h = YahooHistorical()

  # h.download(s)
  h.open(dest_file)
  y = h.get(f, t)
  x = [row['date'] for row in y]
  sma20 = h.moving_average(20, type='simple') # 20 day moving average
  sma200 = h.moving_average(200, type='simple') # 200 day moving average
  rsi = h.relative_strength(f, t)

  plt.rc('axes', grid=True)
  plt.rc('grid', color='0.75', linestyle='-', linewidth=0.5)

  plt.plot(x, [row['adj_close'] for row in y], color='blue', label=s)
  plt.plot(x, sma200[-len(x):], color='red', label='SMA(200)')
  plt.plot(x, sma20[-len(x):], color='green', label='SMA(20)')

  plt.legend()

  plt.show()
