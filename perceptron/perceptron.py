# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:36:34 2015

@author: shimba
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import cross_validation

eta = 1

# Load the digits dataset
digits = datasets.load_digits(2)    # load two classes, 0 and 1
data = digits.data
targets = digits.target
n = len(data)

# add one dimention to future vector for bias
ones = np.ones((n, 1))
X = np.hstack((ones, data))

# put labels into variable y
y = targets.reshape(n)
y[y == 0] = -1  # zero to -1

X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(X, y)

n_train = len(X_train)
n_valid = len(X_valid)

w = np.zeros_like(X[0])

for r in range(5):
    print "iteration", r
    for xi, yi in zip(X_train, y_train):
        if np.sign(np.dot(xi, w)) != yi:
            w += eta * xi * yi
            # print "w", w[0]

    # validation
    xw = np.dot(X_valid, w)
    predicted = np.sign(xw)

    print "error rate", np.sum(predicted != y_valid) / float(n_valid)

# plt.matshow(w[1:].reshape(8, 8), cmap=plt.cm.gray)
# print w[1:].reshape(8, 8)
