# -*- coding: utf-8 -*-
"""
Created on Mon May 11 12:03:56 2015

@author: shimba
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from scipy.misc import logsumexp

n_training = 60000      # The number of training set

# hyper parameters
lr = 0.00000001         # Learning rate
num_iteration = 5     # number of epoch
# initialize weight vector parameters
mu = 0.0                # mean
stddev = 0.5            # standard deviation
# cross validation
n_train_rate = 0.9      # ratio of training and validation data

M = 1000     # number of hidden unit

# Load the digits dataset
# fetch_mldata ... dataname is on mldata.org, data_home
# load 10 classes, from 0 to 9
print 'loading mnist dataset'
mnist = datasets.fetch_mldata('MNIST original')
print 'load done'

data = mnist.data
targets = mnist.target

n_data = len(data)                  # number of all data
n_label = len(set(mnist.target))    # number of labels

# X: feature vectors
# add one dimention to future vector for bias
ones = np.ones((n_data, 1))
X = np.hstack((ones, data))

# t: correct labels
# labels to 1 of K
t = np.zeros((n_data, n_label))
for i, target in enumerate(targets):
    t[i].put(target, 1)

# Split test dataset into training dataset (60000), and test dataset (10000)
X_training = X[:n_training]
t_training = t[:n_training]
X_test = X[n_training:]
t_test = t[n_training:]

X_train, X_valid, t_train, t_valid = cross_validation.train_test_split(
    X_training, t_training, train_size=n_train_rate, random_state=0)

n_train = len(X_train)          # number of training dataset
n_valid = len(X_valid)          # number of validation dataset
n_test = len(X_test)            # number of test dataset
d_feature = len(X_train[-1])    # dimention of feature vector

# initialize weight vector
# each w is vertical vector
np.random.seed(0)
w_1 = stddev * np.random.randn(M, d_feature)
w_2 = stddev * np.random.randn(n_label, M)


# for activation
def tanh(a):
    return np.tanh(a)


# softmax function
# input a = np.dot(xi, w.T)
def softmax(a):
    a_T = a.T
    b = logsumexp(a_T, axis=0)
    return np.exp(a_T - b).T

correct_rates_train = []
correct_rates_valid = []

# initialize correct rate of validation.
# It is to store best scored w
correct_rate_valid_best = 0
w_best_1 = w_1
w_best_2 = w_2

# 'r' means iteration. The name 'r' come from PRML.
for r in range(num_iteration):
    print "iteration", r+1

    # - forward - #
    # -- first layer -- #
    # calculate activation
    a_pred_training = np.dot(X_train, w_1.T)
    # convert with activation function
    z_pred_training = tanh(a_pred_training)

    # -- second layer -- #
    # calculate labels
    y_pred_training = softmax(np.dot(z_pred_training, w_2.T))

    # - backprop - #
    # error at layer 2
    error_2 = (y_pred_training - t_train)
    # error at layer 1
    error_1 = (1 - z_pred_training*z_pred_training) * np.dot(error_2, w_2)
    # gradient at layer 1
    gradient_1 = np.dot(X_train.T, error_1).T
    # gradient at layer 2
    gradient_2 = np.dot(z_pred_training.T, error_2).T

    # momentum

    w_1 -= lr * gradient_1
    w_2 -= lr * gradient_2

    assert not np.any(np.isnan(w_1))
    assert not np.any(np.isnan(w_1))

    print "l2 norm w1", np.linalg.norm(w_1)
    print "l2 norm w2", np.linalg.norm(w_2)

    # calculate error rate of training data
    a_pred_train = np.dot(X_train, w_1.T)
    z_pred_train = tanh(a_pred_train)
    y_pred_train = softmax(np.dot(z_pred_train, w_2.T))

    n_fails_train = np.sum(np.argmax(y_pred_train, axis=1) !=
                           np.argmax(t_train, axis=1))
    correct_rate_train = 1 - (n_fails_train / float(n_train))
    print "[train] cor rate ", correct_rate_train
    correct_rates_train.append(correct_rate_train)

    # calculate error rate of validation data
    a_pred_valid = np.dot(X_valid, w_1.T)
    z_pred_valid = tanh(a_pred_valid)
    y_pred_valid = softmax(np.dot(z_pred_valid, w_2.T))
    n_fails_valid = np.sum(np.argmax(y_pred_valid, axis=1) !=
                           np.argmax(t_valid, axis=1))
    correct_rate_valid = 1 - (n_fails_valid / float(n_valid))
    if correct_rate_valid > correct_rate_valid_best:
        w_1_best = w_1
        w_2_best = w_2
        correct_rate_valid_best = correct_rate_valid
    print "[valid] cor rate ", correct_rate_valid
    correct_rates_valid.append(correct_rate_valid)

# show correct rate of train and valid
plt.figure()
plt.plot(np.arange(num_iteration), np.array(correct_rates_train))
plt.plot(np.arange(num_iteration), np.array(correct_rates_valid))
plt.legend(['train', 'valid'])
plt.show()

# -- test -- #
# calculate error rate of test data
a_pred_test = np.dot(X_test, w_1_best.T)
z_pred_test = tanh(a_pred_test)
y_pred_test = softmax(np.dot(z_pred_test, w_2_best.T))
n_fails_test = np.sum(np.argmax(y_pred_test, axis=1) !=
                      np.argmax(t_test, axis=1))
n_correct_test = n_test - n_fails_test

# -- plot confusion matrix start -- #
mnist_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(n_label)
    plt.xticks(tick_marks, mnist_labels, rotation=45)
    plt.yticks(tick_marks, mnist_labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# To plot confusion matrix, convert format 1 of K to label array
def oneK2label(y):
    result = np.zeros(len(y))
    for i in range(len(y)):
        result[i] = np.argmax(y[i])
    return result

t_test_label = oneK2label(t_test)
y_label = oneK2label(y_pred_test)

# Compute confusion matrix
cm = confusion_matrix(t_test_label, y_label)
np.set_printoptions(precision=2)
print 'Confusion matrix, without normalization'
print cm
plt.figure()
plot_confusion_matrix(cm)
# -- plot confusion matrix end -- #

print "[test] correct rate ", (n_correct_test / float(n_test))
