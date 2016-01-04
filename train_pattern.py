from __future__ import print_function # python3 print function

__author__ = 'Joseph Cheng <indiejoseph@gmail.com>'
__version__ = '0.1'

import os
from sys import stdout
import matplotlib.pyplot as plt
from data_prepare import YahooHistorical
from datetime import date, datetime
import numpy as np
import random
import talib
from lib.features import Features
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import RPropMinusTrainer, BackpropTrainer
from pybrain.datasets import SequentialDataSet
from pybrain.structure.modules import LSTMLayer, SigmoidLayer, LinearLayer, TanhLayer, SoftmaxLayer
from pybrain.structure.modules.biasunit import BiasUnit
from pybrain.structure.connections.full import FullConnection
from pybrain.structure import RecurrentNetwork
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
from pybrain.utilities import percentError
from pybrain.structure.modules import KohonenMap
import pickle

np.random.seed(42)

def normalize(data, mean=None, std=None):
  data = np.asarray(data)
  if mean == None:
    mean = np.mean(data)
  if std == None:
    std = np.std(data)
  return (data - mean) / std, mean, std

symbol1 = '0005.HK'
yahoo_data = YahooHistorical(data_from=date(2000, 1, 1), data_to=date(2015, 12, 31))
yahoo_data.open(os.path.join(os.path.dirname(__file__), 'data/' + symbol1 + '.csv'))
data = yahoo_data.get()
close_prices = np.array([n['adj_close'] for n in data])
# low_prices = np.array([n['low'] for n in data])
# high_prices = np.array([n['high'] for n in data])
# volumes = np.array([n['vol'] for n in data])
# prices, mean, std = normalize(close_prices)
# sma10, mean, std = normalize(talib.SMA(close_prices, 10), mean, std) # 10-day SMA
# sma50, mean, std = normalize(talib.SMA(close_prices, 50), mean, std) # 50-day SMA
# sma200, mean, std = normalize(talib.SMA(close_prices, 200), mean, std) # 200-day SMA
# macd_upper, macd_middle, macd_lower = talib.MACD(close_prices, 12, 26, 9)
# mfi = talib.MFI(high_prices, low_prices, close_prices, volumes) # Money Flow Index
# rsi = talib.RSI(close_prices)
training_input = []
training_output = []
p = 17 # 17-day
nodes = 6
testlen = 300
grid = nodes * nodes

# build network
som = pickle.load(open("pattern.p", "rb"))
net = buildNetwork(grid, grid*3, grid, hiddenclass=LSTMLayer, outclass=SoftmaxLayer, outputbias=False, recurrent=True, bias=True)
# net = NetworkReader.readFrom('pattern_weight.xml')

# preparation
training_dataset = SequenceClassificationDataSet(n_input, 1, nb_classes=3, class_labels=["UP", "DOWN", "NOWHERE"])
for i in xrange(close_prices - testlen):
  training_dataset.newSequence()
  for j in xrange(step, 0, -1):
    xpattern = patterns[i-j]
    ypattern = patterns[i-j+1]
    p_xpattern = som.activate(xpattern)
    p_ypattern = som.activate(ypattern)
    xidx = (p_xpattern[0] * step) + p_xpattern[1]
    yidx = (p_ypattern[0] * step) + p_ypattern[1]
    x = np.zeros(grid)
    y = np.zeros(grid)
    x[xidx] = 1
    y[yidx] = 1
    training_dataset.appendLinked(x, y)

# training
# trainer = RPropMinusTrainer(net, dataset=training_dataset, verbose=True)
## trainer = BackpropTrainer(net, dataset=training_dataset, momentum=0.1, learningrate=0.01, weightdecay=0.05, verbose=True)
# EPOCHS_PER_CYCLE = 10
# CYCLES = 10
# EPOCHS = EPOCHS_PER_CYCLE * CYCLES
# train_errors = [] # save errors for plotting later
# for i in xrange(CYCLES):
#   trainer.trainEpochs(EPOCHS_PER_CYCLE)
#   train_errors.append(trainer.testOnData())
#
#   # export network
#   NetworkWriter.writeToFile(net, 'pattern_weight.xml')

# test
result = []
idx = int(random.uniform(0, len(testing_dataset['input'])-(step+1)))
test_data = testing_dataset['input'][idx:idx+step+1]
print(test_data)
test_input = test_data[:step]
test_target = test_data[-1]
for i in xrange(step):
  output = net.activate(test_input[i])
  if i is (step-1):
    print(np.argmax(test_target))


# render plot
# fig = plt.figure()
# plt.show()
