# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:57:12 2015

@author: shimba
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from scipy.misc import logsumexp

eta = 0.1
n_training = 60000

# Load the digits dataset
# fetch_mldata ... dataname is on mldata.org, data_home
# load 10 classes, from 0 to 9
print 'loading mnist dataset'
mnist = datasets.fetch_mldata('MNIST original', data_home=".")
print 'load done'
data = mnist.data
targets = mnist.target

n = len(data)
n_label = len(set(mnist.target))

# X: feature vectors
# add one dimention to future vector for bias
ones = np.ones((n, 1))
X = np.hstack((ones, data))

# t: correct labels
# labels to 1 of K
t = np.zeros((n, n_label))
for i, j in enumerate(targets):
    t[i].put(j, 1)

# Split test dataset into training dataset (60000), and test dataset (10000)
X_training = X[:n_training]
t_training = t[:n_training]
X_test = X[n_training:]
t_test = t[n_training:]

X_train, X_valid, t_train, t_valid = cross_validation.train_test_split(
    X_training, t_training, random_state=0)

n_train = len(X_train)
n_valid = len(X_valid)
n_test = len(X_test)
d_feature = len(X_train[-1])

# feature vector dimension
D = X_train.shape[-1]

# initialize weight vector
# each w is vertical vector
w = np.random.rand(n_label, D)


# softmax function
# input a = np.dot(xi, w.T)
def softmax(a):
    #print logsumexp(a, axis=0).sha
    a_T = a.T
    b = logsumexp(a_T, axis=0)
    return np.exp(a_T - b).T
    #return np.exp(a - logsumexp(a, axis=0))

eta = 1.0
num_iteration = 500

error_rates_train = []
error_rates_valid = []

# 'r' means iteration. The name 'r' come from PRML.
for r in range(num_iteration):
    print "iteration", r+1
    gradient = np.zeros((n_label, D))    # initialize gradient: 10 x 65

    # TODO: count time and compare with before implementation
    y = softmax(np.dot(X_train, w.T))
    error = (y - t_train)
    gradient = np.dot(error.T, X_train)
    w -= eta * gradient

    assert not np.any(np.isnan(w))

    eta *= 0.9

    print 'l2 norm (w)', np.linalg.norm(w)

    y = softmax(np.dot(X_train, w.T))
    n_fails_train = np.sum(y != t_train) / 2
    error_rate_train = n_fails_train / float(n_train)
    print "[train] error rate", error_rate_train
    error_rates_train.append(error_rate_train)

    y = softmax(np.dot(X_valid, w.T))
    n_fails_valid = np.sum(y != t_valid) / 2
    error_rate_valid = n_fails_valid / float(n_valid)
    print "[valid] error rate", error_rate_valid
    error_rates_valid.append(error_rate_valid)

y = softmax(np.dot(X_test, w.T))
n_fails_test = np.sum(y != t_test) / 2
n_correct_test = n_test - n_fails_test

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 20))
for wk, ax in zip(w, axes.ravel()):
    ax.matshow(wk[1:].reshape(28, 28), cmap=plt.cm.gray)

plt.figure()
plt.plot(np.arange(num_iteration), np.array(error_rates_train))
plt.plot(np.arange(num_iteration), np.array(error_rates_valid))
plt.legend(['train', 'valid'])
plt.show()

print "[test] error rate", n_fails_test / float(n_test)
print "[test] correct rate", n_correct_test / float(n_test)
