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

# X: feature vectors
# add one dimention to future vector for bias
ones = np.ones((n, 1))
X = np.hstack((ones, data))     # 360 x 65

# t: correct labels
t = targets     # 360 x 1

X_train, X_valid, t_train, t_valid = cross_validation.train_test_split(X, t)

n_train = len(X_train)
n_valid = len(X_valid)

# add bias to feature vector dimension
D = X_train.shape[-1]

# initialize weight vector
w = np.random.rand(D)


def sigmoid(a):
    return 1. / (1. + np.exp(-a))


# 'r' means iteration. The name 'r' come from PRML.
for r in range(5):
    print "iteration", r+1
    gradient = 0    # initialize gradient
    for xi, ti in zip(X_train, t_train):
        # y is predicted label
        y = sigmoid(np.dot(w.T, xi))
        gradient += np.dot((y - ti), xi)
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
