# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:57:12 2015

@author: shimba
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from scipy.misc import logsumexp

n_training = 60000

# hyper parameters
lr = 0.00000001       # Learning rate
lr_decay_rate = 1.0
num_iteration = 300    # number of epoch
minibatch_size = 500    # training data size is 50,000
# initialize weight vector parameters
mu = 0.0                # mean
stddev = 0.0            # standard deviation
# cross validation
n_train_rate = 0.9

# Load the digits dataset
# fetch_mldata ... dataname is on mldata.org, data_home
# load 10 classes, from 0 to 9
print('loading mnist dataset')
mnist = datasets.fetch_mldata('MNIST original')
print('load done')
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
for i, target in enumerate(targets):
    t[i].put(target, 1)

# Split test dataset into training dataset (60000), and test dataset (10000)
X_training = X[:n_training]
t_training = t[:n_training]
X_test = X[n_training:]
t_test = t[n_training:]

X_train, X_valid, t_train, t_valid = cross_validation.train_test_split(
    X_training, t_training, train_size=n_train_rate, random_state=0)

n_train = len(X_train)
n_valid = len(X_valid)
n_test = len(X_test)
d_feature = len(X_train[-1])

# feature vector dimension
D = X_train.shape[-1]

np.random.seed(0)
# initialize weight vector
# each w is vertical vector
w = stddev * np.random.randn(n_label, D)


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
w_best = w

# 'r' means iteration. The name 'r' come from PRML.
for r in range(num_iteration):
    print("%3d" % int(r+1), end=" ")
    gradient = np.zeros((n_label, D))    # initialize gradient: 10 x 65

    # ---
    # generate minibatch here
    # ---
    num_batches = n_train / minibatch_size  # 1エポックあたりのミニバッチの個数
    # 添字配列[0, 1, ..., n_train-1] のシャッフル
    perm = np.random.permutation(n_train)
    X_train_batchs = []
    t_train_batchs = []
    # ランダム添字を分割しイテレーション
    for indices in np.array_split(perm, num_batches):
        X_train_batchs.append(X_train[indices])
        t_train_batchs.append(t_train[indices])

    # mini batch SGD training start
    for X_train_batch, t_train_batch in zip(X_train_batchs, t_train_batchs):
        y_training = softmax(np.dot(X_train_batch, w.T))
        error = (y_training - t_train_batch)
        gradient = np.dot(error.T, X_train_batch)
        w -= lr * gradient
        assert not np.any(np.isnan(w))
    # training done

    lr *= lr_decay_rate  # update eta

    print("l2 norm %0.4f" % np.linalg.norm(w), end=" ")

    # calculate error rate of training data
    y_pred_train = softmax(np.dot(X_train, w.T))
    n_fails_train = np.sum(np.argmax(y_pred_train, axis=1) !=
                           np.argmax(t_train, axis=1))
    correct_rate_train = 1 - (n_fails_train / float(n_train))
    print("[train] cor rate %0.4f" % correct_rate_train, end=" ")
    correct_rates_train.append(correct_rate_train)

    # calculate error rate of validation data
    y_pred_valid = softmax(np.dot(X_valid, w.T))
    n_fails_valid = np.sum(np.argmax(y_pred_valid, axis=1) !=
                           np.argmax(t_valid, axis=1))
    correct_rate_valid = 1 - (n_fails_valid / float(n_valid))
    if correct_rate_valid > correct_rate_valid_best:
        w_best = w
        correct_rate_valid_best = correct_rate_valid
    print("[valid] cor rate %0.4f" % correct_rate_valid)
    correct_rates_valid.append(correct_rate_valid)

# plot weight vector
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 20))
for wk, ax in zip(w, axes.ravel()):
    ax.matshow(wk[1:].reshape(28, 28), cmap=plt.cm.gray)

# show correct rate of train and valid
plt.figure()
plt.plot(np.arange(num_iteration), np.array(correct_rates_train))
plt.plot(np.arange(num_iteration), np.array(correct_rates_valid))
plt.legend(['train', 'valid'])
plt.show()

# -- test -- #
# calculate error rate of test data
y_pred_test = softmax(np.dot(X_test, w_best.T))
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
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)
# -- plot confusion matrix end -- #

print("[test] correct rate %f" % (n_correct_test / float(n_test)))
