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


# X is feature vectors
X = data        # 360 x 65

# t is correct labels
t = targets     # 360 x 1


X_train, X_valid, t_train, t_valid = cross_validation.train_test_split(X, t)

n_train = len(X_train)
n_valid = len(X_valid)

# initialize weight vector
w = np.random.rand(64)


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
        y = sigmoid(np.dot(w.T, xi))
        # float to int with round off
        if int(round(y)) == ti:
            count_fails += 1
    print "error rate", count_fails / float(n_valid)
