# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 02:01:07 2015

@author: shimba
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from chainer import Variable, FunctionSet, optimizers
import chainer.functions as F
import draw_filters

batchsize = 20
n_epoch = 15
n_units = 100
D = 784

lr = 0.001
std_w1_init = 0.05
std_w2_init = 0.03

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



# Neural net architecture
def forward(x_data, y_data, train=True):
    x, t = Variable(x_data), Variable(y_data)
    h = F.dropout(F.relu(model.l1(x)), train=train)
    y = model.l2(h)
    return F.mean_squared_error(y, t)


def predict(x_data, train=False):
    x = Variable(x_data)
    h = F.dropout(F.relu(model.l1(x)), train=train)
    y = model.l2(h)
    return y

# Setup optimizer
optimizer = optimizers.Adam(alpha=lr)
optimizer.setup(model.collect_parameters())

# Learning loop
for epoch in xrange(1, n_epoch+1):
    print 'epoch', epoch

    # training
    perm = np.random.permutation(N)
    sum_loss = 0
    for i in xrange(0, N, batchsize):
        x_batch = x_train[perm[i:i+batchsize]]
        y_batch = y_train[perm[i:i+batchsize]]

        optimizer.zero_grads()
        loss = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * batchsize

    print 'train mean loss={}'.format(sum_loss / N)

    print "[w1l2] %5.4f" % np.linalg.norm(model.l1.W)
    print "[w2l2] %5.4f" % np.linalg.norm(model.l2.W)

    print "[g1l2] %5.4f" % np.linalg.norm(model.l1.gW)
    print "[g2l2] %5.4f" % np.linalg.norm(model.l2.gW)

    draw_filters.draw_filters(model.l1.W)
    plt.draw()

result = np.empty(D)
y = predict(x_test)
for i in range(10):
    index = (labels_test == i)
    for j in range(5):
        result = np.vstack((result, x_test[index][j]))
        result = np.vstack((result, y.data[index][j]))

draw_filters.draw_filters(result[1:])
