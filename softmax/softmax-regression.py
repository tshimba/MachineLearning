# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:46:58 2015

@author: shimba
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import cross_validation

eta = 0.1

# Load the digits dataset
digits = datasets.load_digits()     # load 10 classes, from 0 to 9
data = digits.data                  # (1797L, 64L)
targets = digits.target             # (1797L,)

n = len(data)   # 1797, integer
n_label = len(digits.target_names)  # Number of labels

# X: feature vectors
# add one dimention to future vector for bias
ones = np.ones((n, 1))
X = np.hstack((ones, data))     # (1797L, 65L)

# t: correct labels
# labels to 1 of K
t = np.zeros((n, n_label))      # (1797L, 10L)
for i, j in enumerate(targets):
    t[i].put(j, 1)

X_train, X_valid, t_train, t_valid = cross_validation.train_test_split(X, t)

n_train = len(X_train)
n_valid = len(X_valid)

# feature vector dimension
D = X_train.shape[-1]

# initialize weight vector
# each w is vertical vector
w = np.random.rand(n_label, D)  # 10 x 65


# softmax function
# input a = np.dot(X[0], w.T)
def softmax(a):
    return np.exp(a) / np.sum(np.exp(a))


# 'r' means iteration. The name 'r' come from PRML.
for r in range(5):
    print "iteration", r+1
    gradient = np.zeros((n_label, D))    # initialize gradient: 10 x 65
    for xi, ti in zip(X_train, t_train):
        # y is predicted label
        y = softmax(np.dot(xi, w.T))  # y.shape -> (10L,)
        error = y - ti
        for i in range(n_label):
            gradient[i] += error[i] * xi
    w -= eta * gradient
    print w

    #count_fails = 0
    #for xi, ti in zip(X_valid, t_valid):  # validation data
        #print softmax(np.dot(xi, w.T))
        #time.sleep(0.1)
