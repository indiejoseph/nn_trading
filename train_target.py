from __future__ import print_function # python3 print function

__author__   = 'Joseph Cheng <indiejoseph@gmail.com>'
__version__  = '0.1'

import os
import pickle
from tabulate import tabulate
from data_prepare import YahooHistorical
from datetime import date, datetime
import numpy as np
from itertools import cycle
from sys import stdout
import matplotlib.pyplot as plt
from scipy import random
from scipy.signal import argrelextrema
from pybrain.supervised import RPropMinusTrainer, BackpropTrainer
from pybrain.datasets import SequentialDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer, SigmoidLayer, LinearLayer, TanhLayer, SoftmaxLayer
from pybrain.tools.validation import testOnSequenceData
from pybrain.structure.modules.neuronlayer import NeuronLayer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader

random.seed(42)

def ir(p, i, data): # p-days, i=current day
  a = np.sum([n['adj_close'] for n in data[i:i+p]]) / p
  c = data[i]['adj_close']
  return ((a - c) / c)

# prepare date
symbol = '^HSI'
yahoo_data = YahooHistorical()
yahoo_data.open(os.path.join(os.path.dirname(__file__), 'data/' + symbol + '.csv'))
training_set = np.array([n for n in yahoo_data.data if n['date'] >= date(2007, 1, 1) and n['date'] <= date(2013, 12, 31)])
test_set = np.array([n for n in yahoo_data.data if n['date'] >= date(2014, 1, 1) and n['date'] <= date(2015, 12, 31)])
rsi = yahoo_data.relative_strength(n=14)
(sma13, sma7, macd) = yahoo_data.moving_average_convergence(7, 13) # 7 days and 13 days moving average and MACD
test_label = []
training_label = [n['date'] for n in training_set]
training_list = np.array([n['adj_close'] for n in training_set])
training_target = np.zeros(len(training_list))
test_list = np.array([n['adj_close'] for n in test_set])
test_target = np.zeros(len(test_list))
test_target[list(argrelextrema(test_list, np.greater)[0])] = 1
test_target[list(argrelextrema(test_list, np.less)[0])] = -1
training_target[list(argrelextrema(training_list, np.greater)[0])] = 1
training_target[list(argrelextrema(training_list, np.less)[0])] = -1
(tmax50, tmin50) = yahoo_data.trading_range_breakout(50)
(tmax100, tmin100) = yahoo_data.trading_range_breakout(100)
(tmax200, tmin200) = yahoo_data.trading_range_breakout(200)
(training_list, tmean, tstd) = normalize([[n['open'], n['high'], n['low'], n['adj_close']] for n in training_set])
(test_list, tmean, tstd) = normalize([[n['open'], n['high'], n['low'], n['adj_close']] for n in test_set], tmean, tstd)
(tmax50, tmean, tstd) = normalize(tmax50, tmean, tstd)
(tmin50, tmean, tstd) = normalize(tmin50, tmean, tstd)
(tmax100, tmean, tstd) = normalize(tmax100, tmean, tstd)
(tmin100, tmean, tstd) = normalize(tmin100, tmean, tstd)
(tmax200, tmean, tstd) = normalize(tmax200, tmean, tstd)
(tmin200, tmean, tstd) = normalize(tmin200, tmean, tstd)
(sma7, s20mean, s20std) = normalize(sma7, tmean, tstd)
(sma13, s100mean, s100std) = normalize(sma13, tmean, tstd)
(macd, macdmean, macdstd) = normalize(macd)
(rsi, rsimean, rsistd) = normalize(rsi)
dlen = len(training_list)

# plt.grid(True)
# plt.plot(training_label, [n[3] for n in training_list], color="blue", label=symbol)
# plt.plot(training_label, rsi[-dlen:], color=random.rand(3,1), label='RSI')
# plt.plot(training_label, macd[-dlen:], color=random.rand(3,1), label='MACD')
# plt.plot(training_label, rsi[-dlen:], color=random.rand(3,1), label='RSI')
# plt.plot(training_label, sma7[-dlen:], color=random.rand(3,1), label='EMA7')
# plt.plot(training_label, sma13[-dlen:], color=random.rand(3,1), label='EMA13')
# plt.plot(training_label, tmax50[-dlen:], color=random.rand(3,1), label='tmax50')
# plt.plot(training_label, tmin50[-dlen:], color=random.rand(3,1), label='tmin50')
# plt.plot(training_label, tmax100[-dlen:], color=random.rand(3,1), label='tmax100')
# plt.plot(training_label, tmin100[-dlen:], color=random.rand(3,1), label='tmin100')
# plt.plot(training_label, tmax200[-dlen:], color=random.rand(3,1), label='tmax200')
# plt.plot(training_label, tmin200[-dlen:], color=random.rand(3,1), label='tmin200')
# plt.legend()
# plt.show()

# build network
# net = buildNetwork(14, 54, 1, hiddenclass=LSTMLayer, outclass=TanhLayer, outputbias=False, recurrent=True)
net = NetworkReader.readFrom('target_weight.xml')

net.randomize()

# build sequential dataset
train_ds = SequentialDataSet(14, 1)

for n, r, m, m7, m13, tx50, tn50, tx100, tn100, tx200, tn200, target in zip(training_list, rsi[-dlen:], macd[-dlen:], sma7[-dlen:], sma13[-dlen:], tmax50[-dlen:], tmin50[-dlen:], tmax100[-dlen:], tmin100[-dlen:], tmax200[-dlen:], tmin200[-dlen:], training_target):
  i = np.append(n, [r, m, m7, m13, tx50, tn50, tx100, tn100, tx200, tn200])
  train_ds.addSample(i, target)

# train network
trainer = RPropMinusTrainer(net, dataset=train_ds)
# trainer = BackpropTrainer(net, dataset=train_ds)
train_errors = [] # save errors for plotting later
EPOCHS_PER_CYCLE = 100
CYCLES = 20

EPOCHS = EPOCHS_PER_CYCLE * CYCLES
for i in xrange(CYCLES):
  trainer.trainEpochs(EPOCHS_PER_CYCLE)
  train_errors.append(trainer.testOnData())
  epoch = (i+1) * EPOCHS_PER_CYCLE
  print("\r epoch {}/{}".format(epoch, EPOCHS), end="")
  stdout.flush()

# test output
sell_list = []
buy_list = []
sell_label = []
buy_label = []
actual_list = []
IRs = { 7: 0, 5: 0, 3: 0 }
IRb = { 7: 0, 5: 0, 3: 0 }
tlen = len(test_list)

for i, (n, r, m, m7, m13, tx50, tn50, tx100, tn100, tx200, tn200) in enumerate(zip(test_list, rsi[-tlen:], macd[-tlen:], sma7[-tlen:], sma13[-tlen:], tmax50[-tlen:], tmin50[-tlen:], tmax100[-tlen:], tmin100[-tlen:], tmax200[-tlen:], tmin200[-tlen:])):
  date = test_set[i]['date']
  o = net.activate([n[0], n[1], n[2], n[3], r, m, m7, m13, tx50, tn50, tx100, tn100, tx200, tn200])

  if o >= 0.99:
    sell_label.append(date)
    sell_list.append(n[3])
    for p in [7, 5, 3]:
      IRs[p] += ir(p, i, test_set)
  elif o <= -0.99:
    buy_label.append(date)
    buy_list.append(n[3])
    for p in [7, 5, 3]:
      IRb[p] += ir(p, i, test_set)

  test_label.append(date)
  actual_list.append(n[3])

IRba = ((IRb[7] + IRb[5] + IRb[3]) / 3) * 100
IRsa = ((IRs[7] + IRs[5] + IRs[3]) / 3) * 100

print("\r")
print(tabulate([
  ['IRb(%)', (IRb[7] * 100), (IRb[5] * 100), (IRb[3] * 100), IRba],
  ['IRs(%)', (IRs[7] * 100), (IRs[5] * 100), (IRs[3] * 100), IRsa],
  ['Gain(%)', IRba + IRsa, '', '', '']
], headers=['', 'P = 7', 'P = 5', 'P = 3', 'average']))

# export network
NetworkWriter.writeToFile(net, 'target_weight.xml')

# draw plot
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(range(0, EPOCHS, EPOCHS_PER_CYCLE), train_errors)
ax1.set_title('Error')
ax2 = fig.add_subplot(212)
ax2.grid(True)
ax2.set_title('Predict')
ax2.plot(test_label, actual_list, color="blue", label="Acutal")
ax2.fill_between(test_label, actual_list, -1, color="blue", alpha=.15)
ax2.plot(test_label, sma7[-len(test_label):], color="cyan", label="EMA(20)")
ax2.plot(test_label, sma13[-len(test_label):], color="red", label="EMA(100)")
ax2.plot(test_label, macd[-len(test_label):], color="orange", label="MACD")
ax2.plot(test_label, rsi[-len(test_label):], color="pink", label="RSI")
ax2.plot(test_label, tmax50[-len(test_label):], color="tomato", label="TRBx50")
ax2.plot(test_label, tmin50[-len(test_label):], color="tan", label="TRBn50")
ax2.plot(test_label, tmax100[-len(test_label):], color="olive", label="TRBx100")
ax2.plot(test_label, tmin100[-len(test_label):], color="deeppink", label="TRBn100")
ax2.plot(test_label, tmax200[-len(test_label):], color="maroon", label="TRBx200")
ax2.plot(test_label, tmin200[-len(test_label):], color="skyblue", label="TRBn200")
plt.scatter(sell_label, sell_list, color="b", marker="o", label="Sell")
plt.scatter(buy_label, buy_list, color="r", marker="*", label="Buy")
ax2.legend()

plt.show()
