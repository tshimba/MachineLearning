# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:29:56 2015

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

# add bias to feature vector dimension
D = data.shape[-1] + 1

# X: feature vectors
# add one dimension to future vector for bias
ones = np.ones((n, 1))
X = np.hstack((ones, data)).reshape(n, D, 1)     # 360 x 65 x 1

# t: correct labels
t = targets     # (360, ) is 1 dimension.

X_train, X_valid, t_train, t_valid = cross_validation.train_test_split(X, t)

n_train = len(X_train)
n_valid = len(X_valid)

# initialize weight vector
w = np.random.rand(D, 1)    # (65, 1) is 2 dimension


def sigmoid(a):
    return 1. / (1. + np.exp(-a))


# 'r' means iteration. The name 'r' come from PRML.
for r in range(5):
    print "iteration", r+1
    gradient = 0    # initialize gradient
    for xi, ti in zip(X_train, t_train):
        # y is predicted label
        y = sigmoid(np.dot(w.T, xi))
        gradient += (y - ti) * xi
    w = w - eta * gradient

    count_fails = 0
    for xi, ti in zip(X_valid, t_valid):  # validation data
        if np.dot(w.T, xi) > 0:     # sigmoid(0) equals 0.5
            y = 1
        else:
            y = 0
        if y != ti:
            count_fails += 1
    print "error rate", count_fails / float(n_valid)
