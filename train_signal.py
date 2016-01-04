from __future__ import print_function # python3 print function

__author__ = 'Joseph Cheng <indiejoseph@gmail.com>'
__version__ = '0.1'

import os
from sys import stdout
from data_prepare import YahooHistorical
from datetime import date, datetime
import matplotlib.pyplot as plt
import numpy as np
from lib.features import Features
from pybrain.supervised import RPropMinusTrainer, BackpropTrainer
from pybrain.datasets import SequentialDataSet, SequenceClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer, SigmoidLayer, LinearLayer, TanhLayer, SoftmaxLayer
from pybrain.structure.modules.biasunit import BiasUnit
from pybrain.structure.connections.full import FullConnection
from pybrain.structure import RecurrentNetwork
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
from pybrain.utilities import percentError

np.random.seed(42)

symbol = '^HSI'
yahoo_data = YahooHistorical(data_from=date(2001, 1, 1), data_to=date(2015, 12, 31))
yahoo_data.open(os.path.join(os.path.dirname(__file__), 'data/' + symbol + '.csv'))
data = yahoo_data.get()
date = np.asarray([n['date'] for n in data])
test_n = 201
open_prices = np.asarray([n['open'] for n in data])
low_prices = np.asarray([n['low'] for n in data])
high_prices = np.asarray([n['high'] for n in data])
close_prices = np.asarray([n['adj_close'] for n in data])
volumes = np.asarray([n['vol'] for n in data])
features = Features(open_prices, low_prices, high_prices, close_prices, volumes)
f_input = features.getInput()
f_output = features.getOutput()
training_input = f_input[200:-test_n]
training_output = f_output[200:-test_n]
testing_input = f_input[-test_n:-20]
testing_output = f_output[-test_n:-20]
testing_label = date[-test_n:-20]
n_input = len(f_input[0])
n_output = 1

# build sequential dataset
training_dataset = SequenceClassificationDataSet(n_input, n_output, nb_classes=3, class_labels=["UP", "DOWN", "NOWHERE"])
for x, y in zip(training_input, training_output):
  training_dataset.appendLinked(x, [y])
  training_dataset.newSequence()

testing_dataset = SequenceClassificationDataSet(n_input, n_output, nb_classes=3, class_labels=["UP", "DOWN", "NOWHERE"])
for x, y in zip(testing_input, testing_output):
  testing_dataset.appendLinked(x, [y])
  testing_dataset.newSequence()

training_dataset._convertToOneOfMany()
testing_dataset._convertToOneOfMany()

# build network
net = RecurrentNetwork()
net.addInputModule(LinearLayer(training_dataset.indim, name="input"))
net.addModule(LSTMLayer(100, name="hidden1"))
net.addModule(SigmoidLayer(training_dataset.outdim * 3, name="hidden2"))
net.addOutputModule(LinearLayer(training_dataset.outdim, name="output"))
net.addModule(BiasUnit('bias'))
net.addConnection(FullConnection(net["input"], net["hidden1"], name="c1"))
net.addConnection(FullConnection(net["hidden1"], net["hidden2"], name="c3"))
net.addConnection(FullConnection(net["bias"], net["hidden2"], name="c4"))
net.addConnection(FullConnection(net["hidden2"], net["output"], name="c5"))
net.addRecurrentConnection(FullConnection(net["hidden1"], net["hidden1"], name="c6"))
net.sortModules()
# net = buildNetwork(n_input, 256, n_output, hiddenclass=LSTMLayer, outclass=TanhLayer, outputbias=False, recurrent=True)
# net = NetworkReader.readFrom('signal_weight.xml')

# train network
trainer = RPropMinusTrainer(net, dataset=training_dataset, verbose=True, weightdecay=0.01)
# trainer = BackpropTrainer(net, dataset=training_dataset, learningrate = 0.04, momentum = 0.96, weightdecay = 0.02, verbose = True)

for i in range(100):
  # train the network for 1 epoch
  trainer.trainEpochs(5)

  # evaluate the result on the training and test data
  trnresult = percentError(trainer.testOnClassData(), training_dataset['class'])
  tstresult = percentError(trainer.testOnClassData(dataset=testing_dataset), testing_dataset['class'])

  # print the result
  print("epoch: %4d" % trainer.totalepochs, \
        "  train error: %5.2f%%" % trnresult, \
        "  test error: %5.2f%%" % tstresult)
  if tstresult <= 0.5 :
       print('Bingo !!!!!!!!!!!!!!!!!!!!!!')
       break

  # export network
  NetworkWriter.writeToFile(net, 'signal_weight.xml')

# run test
actual_price = np.array([n[3] for n in testing_input])
predict_short = []
predict_long = []
result_long = []
result_short = []

for i, (x, y) in enumerate(zip(testing_input, testing_output)):
  z = net.activate(x)
  predict_short.append(z[0])
  predict_long.append(z[1])
  result_long.append(abs(testing_output[0] - z[0]))
  result_short.append(abs(testing_output[1] - z[1]))

predict_short = np.asarray(predict_short)
predict_long = np.asarray(predict_long)
short_up_idxs = predict_short > 0
short_down_idxs = predict_short < 0
long_up_idxs = predict_long > 0
long_down_idxs = predict_long < 0

# print test
def normalize_result(data):
  max_v = np.max(data)
  min_v = np.min(data)
  p = (data - min_v) / (max_v - min_v)
  return np.sum(p) / len(data)

print("\n")
print("Long test: ", normalize_result(result_long))
print("Short test: ", normalize_result(result_short))

# draw plot
fig = plt.figure()
ax1 = fig.add_subplot(212)
ax1.grid(True)
ax1.set_title('Predict Short')
ax1.plot(testing_label, actual_price, color="blue", label=symbol)
ax1.fill_between(testing_label, actual_price, np.min(actual_price), color="blue", alpha=.15)
ax1.scatter(testing_label[short_up_idxs], actual_price[short_up_idxs], color="g", marker="o", label="Short Up")
ax1.scatter(testing_label[short_down_idxs], actual_price[short_down_idxs], color="r", marker="*", label="Short Down")
ax1.legend()
ax2 = fig.add_subplot(213)
ax2.grid(True)
ax2.set_title('Predict Long')
ax2.plot(testing_label, actual_price, color="blue", label=symbol)
ax2.fill_between(testing_label, actual_price, np.min(actual_price), color="blue", alpha=.15)
ax2.scatter(testing_label[long_up_idxs], actual_price[long_up_idxs], color="g", marker="o", label="Long Up")
ax2.scatter(testing_label[long_down_idxs], actual_price[long_down_idxs], color="r", marker="*", label="Long Down")
ax2.legend()
plt.show()
