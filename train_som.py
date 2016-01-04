from __future__ import print_function # python3 print function

__author__ = 'Joseph Cheng <indiejoseph@gmail.com>'
__version__ = '0.1'

import os
from sys import stdout
import matplotlib.pyplot as plt
from data_prepare import YahooHistorical
from datetime import date, datetime
import numpy as np
from lib.features import Features
from pybrain.structure.modules import KohonenMap
import pickle

np.random.seed(42)

symbol1 = '0005.HK'
yahoo_data1 = YahooHistorical(data_from=date(2000, 1, 1), data_to=date(2015, 12, 31))
yahoo_data1.open(os.path.join(os.path.dirname(__file__), 'data/' + symbol1 + '.csv'))
data1 = yahoo_data1.get()
dataset1 = np.asarray([n['adj_close'] for n in data1])
p = 17 # 17-day
p = 5 # 5-day
nodes = 3
som = KohonenMap(p, nodes)
# som = pickle.load(open("pattern5.p", "rb"))
som.learningrate = 0.01
epochs = 1000
training_dataset = []
result = {}

# preparation
for i in xrange(p, len(dataset1)):
  training_input = dataset1[i-p:i]
  mmax = np.max(training_input)
  mmin = np.min(training_input)
  training_input = (training_input - mmin) / (mmax - mmin)
  if np.isnan(training_input).any():
    training_input = np.array([0] * p)
  training_dataset.append(training_input)

# training
for epoch in xrange(epochs):
  for j in xrange(len(training_dataset)):
    training_input = training_dataset[j]
    output = som.activate(training_input)
    som.backward()
  print(epoch)

# export network
pickle.dump(som, open("pattern5.p", "wb"))

# test
for i in xrange(len(training_dataset)):
  x = training_dataset[i]
  y = som.activate(x)
  key = ','.join([str(int(f)) for f in y.tolist()])

  if key not in result:
    result[key] = []

  result[key].append(x)

print('Category: ', result.keys())

# render plot
fig = plt.figure()
idx = 0
for key, val in result.iteritems():
  idx += 1
  ax = fig.add_subplot(nodes, nodes, idx)
  ax.set_title(key)
  val = zip(*val)
  ax.plot(range(0, p), val, color="#dfdfdf", label=key)
  ax.plot(range(0, p), np.mean(val, axis=1), color="b", label='AVG')

plt.show()
