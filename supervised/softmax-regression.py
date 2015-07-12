# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:46:58 2015

@author: shimba
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import cross_validation
from scipy.misc import logsumexp
import my_logsumexp

# Load the digits dataset
digits = datasets.load_digits()     # load 10 classes, from 0 to 9
data = digits.data                  # (1797L, 64L)
targets = digits.target             # (1797L,)

n = len(data)   # 1797, integer
n_label = len(set(digits.target))  # Number of labels

# X: feature vectors
# add one dimention to future vector for bias
ones = np.ones((n, 1))
X = np.hstack((ones, data))     # (1797L, 65L)

# t: correct labels
# labels to 1 of K
t = np.zeros((n, n_label))      # (1797L, 10L)
t[range(n), targets] = 1

X_train, X_valid, t_train, t_valid = cross_validation.train_test_split(
    X, t, test_size=0.5)

n_train = len(X_train)
n_valid = len(X_valid)

# feature vector dimension
D = X_train.shape[-1]


# softmax function
# input a = np.dot(xi, w.T)
def softmax(a):
    return np.exp(a - my_logsumexp.my_logsumexp(a))


# initialize weight vector
# each w is vertical vector
w = np.random.rand(n_label, D)  # 10 x 65

eta = 0.1
num_iteration = 200

error_rates_train = []
error_rates_valid = []

# 'r' means iteration. The name 'r' come from PRML.
for r in range(num_iteration):
    print("iteration", r+1)
    gradient = np.zeros((n_label, D))    # initialize gradient: 10 x 65

    '''
    y = softmax(np.dot(X_train, w.T))
    error = (y - t_train).reshape(n_label, n)
    gradient = error * X_train

    '''
    for xi, ti in zip(X_train, t_train):
        # y is predicted label
        y = softmax(np.dot(xi, w.T))  # y.shape -> (10L,)
        error = (y - ti).reshape(n_label, 1)
        gradient += error * xi
        assert not np.any(np.isnan(gradient))
    w -= eta * gradient
    assert not np.any(np.isnan(w))

    print('l2 norm (w)', np.linalg.norm(w))

    count_fails = 0
    for xi, ti in zip(X_train, t_train):  # validation data
        y = softmax(np.dot(xi, w.T))
        if np.argmax(y) != np.argmax(ti):
            count_fails += 1
    error_rate_train = count_fails / float(n_train)
    error_rates_train.append(error_rate_train)
    print("[train] error rate", error_rate_train)

    count_fails = 0
    for xi, ti in zip(X_valid, t_valid):  # validation data
        y = softmax(np.dot(xi, w.T))
        if np.argmax(y) != np.argmax(ti):
            count_fails += 1
    error_rate_valid = count_fails / float(n_valid)
    error_rates_valid.append(error_rate_valid)
    print("[train] error rate", error_rate_valid)

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 20))
for wk, ax in zip(w, axes.ravel()):
    ax.matshow(wk[1:].reshape(8, 8), cmap=plt.cm.gray)

plt.figure()
plt.plot(np.arange(num_iteration), np.array(error_rates_train))
plt.plot(np.arange(num_iteration), np.array(error_rates_valid))
plt.legend(['train', 'valid'])
plt.show()
