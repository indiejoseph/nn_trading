from __future__ import print_function # python3 print function

__author__   = 'Joseph Cheng <indiejoseph@gmail.com>'
__version__  = '0.1'

import os
from data_prepare import YahooHistorical
from datetime import date, datetime
import numpy as np
from itertools import cycle
from sys import stdout
import matplotlib.pyplot as plt
from pybrain.supervised import RPropMinusTrainer, BackpropTrainer
from pybrain.datasets import SequentialDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer, SigmoidLayer, LinearLayer, TanhLayer
from pybrain.tools.validation import testOnSequenceData

# prepare date
yahoo_data = YahooHistorical()
yahoo_data.open(os.path.join(os.path.dirname(__file__), 'data/^HSI.csv'))
dataset = yahoo_data.get()

# build network
net = buildNetwork(5, 25, 2, hiddenclass=LSTMLayer, outclass=SigmoidLayer, outputbias=False, recurrent=True)
net.randomize()

# build sequential dataset
train_ds = SequentialDataSet(5, 2)
for n, n1, m20 in zip(training_set, training_set[1:], sma20[-len(training_set):]):
  i = [n['open'], n['high'], n['low'], n['adj_close'], m20]
  d = (n1['adj_close'] - n['adj_close']) / n['adj_close']
  o = [-1, -1]
  if d > 0:
    o[0] = abs(d)
  else:
    o[1] = abs(d)
  train_ds.addSample(i, o)

# build test dataset
test_ds = SequentialDataSet(5, 2)
for n, n1, m20 in zip(test_set, test_set[1:], sma20[-len(test_set):]):
  test_label.append(n['date'])
  i = [n['open'], n['high'], n['low'], n['adj_close'], m20]
  d = (n1['adj_close'] - n['adj_close']) / n['adj_close']
  o = [0, 0]
  if d > 0:
    o[0] = abs(d)
  else:
    o[1] = abs(d)
  test_ds.addSample(i, o)

# train network
trainer = RPropMinusTrainer(net, dataset=train_ds)
# trainer = BackpropTrainer(net, dataset=train_ds)
train_errors = [] # save errors for plotting later
EPOCHS_PER_CYCLE = 100
CYCLES = 10

EPOCHS = EPOCHS_PER_CYCLE * CYCLES
for i in xrange(CYCLES):
  trainer.trainEpochs(EPOCHS_PER_CYCLE)
  train_errors.append(trainer.testOnData())
  epoch = (i+1) * EPOCHS_PER_CYCLE
  print("\r epoch {}/{}".format(epoch, EPOCHS), end="")
  stdout.flush()

# test output
predict_list = []
actual_list = []
err = []

for sample, target in test_ds.getSequenceIterator(0):
  p = net.activate(sample)
  a = np.argmax(p)
  d1 = p[a]
  if a == 1:
    d1 = -p[a]
  t = np.argmax(target)
  d2 = target[t]
  if t == 1:
    d2 = -target[t]
  predict = (d1 * sample[3]) + sample[3]
  actual = (d2 * sample[3]) + sample[3]
  err.append(abs((d2 - d1) / d1)) # compute error rate
  predict_list.append(predict)
  actual_list.append(actual)

test_label = test_label[1:]
actual_list = actual_list[1:]
predict_list = predict_list[1:]

print('\n e:', np.mean(err), '%') # display error rate

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(range(0, EPOCHS, EPOCHS_PER_CYCLE), train_errors)
ax1.set_title('Error')

ax2 = fig.add_subplot(212)
ax2.grid(True)
ax2.set_title('Predict')
ax2.plot(test_label, actual_list, color="blue", label="Acutal")
ax2.fill_between(test_label, actual_list, 0, color="blue", alpha=.15)
ax2.plot(test_label, predict_list, color="green", label="Predict")
ax2.plot(test_label, sma20[-len(test_label):], color="cyan", label="MA(20)")
ax2.legend()

plt.show()
