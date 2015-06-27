# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 02:01:07 2015

@author: shimba
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn import cross_validation
from chainer import Variable, FunctionSet, optimizers
import chainer.functions as F
import draw_filters

batchsize = 20
n_epoch = 100
n_units = 100
D = 784

lr = 0.01
std_w1_init = 1.0
std_w2_init = 0.2
regularization = 0.001

N_train_rate = 0.9

np.random.seed(0)


def generate_noisy_data(data, stddev=1.0):
    return data + stddev * np.random.randn(data.shape[0], data.shape[1])

# Prepare dataset
print 'fetch MNIST dataset'
mnist = fetch_mldata('MNIST original')
mnist.data = mnist.data.astype(np.float32)
data = mnist.data / 255
target = data
labels_test = mnist.target[60000:]

N = 60000
x_train, x_test = np.split(data, [N])
y_train, y_test = np.split(target, [N])
N_test = y_test.size

# Prepare multi-layer perceptron model
model = FunctionSet(l1=F.Linear(D, n_units, wscale=std_w1_init),
                    l2=F.Linear(n_units, D, wscale=std_w2_init))

# holdout validation
x_train, x_valid, y_train, y_valid = \
    cross_validation.train_test_split(x_train, y_train,
                                      train_size=N_train_rate,
                                      random_state=0)

N_train = len(x_train)


# Neural net architecture
def forward(x_data, y_data):
    x, t = Variable(x_data), Variable(y_data)
    h = F.sigmoid(model.l1(x))
    y = F.sigmoid(model.l2(h))
    return F.mean_squared_error(y, t)


def predict(x_data, y_data):
    x, t = Variable(x_data), Variable(y_data)
    h = F.sigmoid(model.l1(x))
    y = F.sigmoid(model.l2(h))
    return y, F.mean_squared_error(y, t)

# Setup optimizer
optimizer = optimizers.Adam(alpha=lr)
optimizer.setup(model.collect_parameters())

scores_train = []
scores_valid = []

# Learning loop
for epoch in xrange(1, n_epoch+1):
    print 'epoch', epoch

    x_train_noise = generate_noisy_data(x_train, stddev=stddev)
    x_valid_noise = generate_noisy_data(x_valid, stddev=stddev)

    # training
    perm = np.random.permutation(N_train)

    for i in xrange(0, N_train, batchsize):
        x_batch = x_train_noise[perm[i:i+batchsize]]
        y_batch = y_train[perm[i:i+batchsize]]

        optimizer.zero_grads()
        loss = forward(x_batch, y_batch)
        loss.backward()
        optimizer.weight_decay(regularization)
        optimizer.update()

    print "[w1l2] %5.4f" % np.linalg.norm(model.l1.W)
    print "[w2l2] %5.4f" % np.linalg.norm(model.l2.W)

    print "[g1l2] %5.4f" % np.linalg.norm(model.l1.gW)
    print "[g2l2] %5.4f" % np.linalg.norm(model.l2.gW)

    y, score = predict(x_train_noise, y_train)
    print "[train]", score.data
    scores_train.append(float(score.data))

    y, score = predict(x_valid_noise, y_valid)
    print "[valid]", score.data
    scores_valid.append(float(score.data))

    # show error rate of train and valid
    plt.figure()
    plt.plot(np.arange(len(scores_train)), np.array(scores_train))
    plt.plot(np.arange(len(scores_valid)), np.array(scores_valid))
    plt.legend(['train', 'valid'])
    plt.show()

    draw_filters.draw_filters(model.l1.W)
    plt.draw()

# show error rate of train and valid
plt.figure()
plt.plot(np.arange(len(scores_train)), np.array(scores_train))
plt.plot(np.arange(len(scores_valid)), np.array(scores_valid))
plt.legend(['train', 'valid'])
plt.show()

x_test = generate_noisy_data(x_test, stddev=stddev)
result = np.empty(D)
y, score = predict(x_test, y_test)
for i in range(10):
    index = (labels_test == i)
    for j in range(5):
        result = np.vstack((result, x_test[index][j]))
        result = np.vstack((result, y.data[index][j]))

draw_filters.draw_filters(result[1:])
