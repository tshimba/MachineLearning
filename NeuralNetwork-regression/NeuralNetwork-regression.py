# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:40:24 2015

@author: shimba
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from scipy.misc import logsumexp
import time


def softmax(a):
    '''
    softmax function
    Usage:
    input a is np.dot(xi, w.T)
    '''
    a_T = a.T
    b = logsumexp(a_T, axis=0)
    return np.exp(a_T - b).T


def label_to_onehot(labels):
    '''
    t: correct labels
    labels to 1 of K
    '''
    n_examples = len(labels)
    n_classes = len(np.unique(labels))
    onehot = np.zeros((n_examples, n_classes), dtype=np.float32)
    for i, label in enumerate(labels):
        onehot[i].put(label, 1)
    return onehot


def generate_noisy_sin(num_examples=1000, noise_std=0.2):
    x = np.random.uniform(-10, 10, num_examples)
    y_true = np.sin(x)
    y = y_true + noise_std * np.random.randn(num_examples)
    return x, y


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, mnist_labels, rotation=45)
    plt.yticks(tick_marks, mnist_labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


class NeuralNetworkClassifier(object):
    def __init__(self, M=100):
        self.M = M
        self.w_1 = None
        self.w_2 = None
        self.v_1 = None
        self.v_2 = None

    def fit(self, data_train, label_train, data_valid, label_valid,
            lr=0.001, num_iteration=20, minibatch_size=500, mc=0.0,
            regularization=0.0, std_w1_init=0.5, std_w2_init=0.2):

        n_classes = len(np.unique(label_train))
        n_train = len(data_train)              # number of training dataset
        d_feature = len(data_train[-1]) + 1    # dimention of feature vector

        # initialize weight vector
        # each w is vertical vector
        np.random.seed(0)
        self.w_1 = std_w1_init * np.random.randn(self.M,
                                                 d_feature).astype(np.float32)
        self.w_1[:, 0] = 0
        # 1 for bias at hidden layer
        self.w_2 = std_w2_init * np.random.randn(n_classes,
                                                 self.M+1).astype(np.float32)
        self.w_2[:, 0] = 0

        self.v_1 = 0
        self.v_2 = 0

        scores_train = []
        scores_valid = []

        # initialize correct rate of validation.
        # It is to store best scored w
        score_valid_best = 0

        # t convert to 1 hot
        t_train = label_to_onehot(label_train)

        # X: feature vectors
        # add one dimention to future vector for bias
        n_train = len(data_train)                  # number of all data

        ones = np.ones((n_train, 1), dtype=np.float32)
        X_train = np.hstack((ones, data_train)).astype(np.float32)

        try:
            # 'r' means iteration. The name 'r' come from PRML.
            for r in range(num_iteration):
                measure_start = time.clock()
                print "iteration %3d" % (r+1)

                # numbers of minibatch per 1 epoch
                num_batches = n_train / minibatch_size
                # shuffle the list of indices
                perm = np.random.permutation(n_train)
                # separate randomized indices and loop processing
                # mini batch SGD training
                for indices in np.array_split(perm, num_batches):
                    X_batch = X_train[indices]
                    t_batch = t_train[indices]
                    # - forward pass - #
                    # -- first layer -- #
                    # calculate activation
                    a = np.dot(X_batch, self.w_1.T)
                    # convert with activation function
                    z = np.tanh(a)
                    # add bias to z_pred_training
                    ones = np.ones((minibatch_size, 1), dtype=np.float32)
                    z = np.hstack((ones, z))

                    # -- second layer -- #
                    # calculate labels
                    y = softmax(np.dot(z, self.w_2.T))

                    # - backward pass - #
                    # error at layer 2
                    error_2 = (y - t_batch)
                    # error at layer 1
                    error_1 = (1 - z[:, 1:] ** 2) * np.dot(error_2,
                                                           self.w_2[:, 1:])
                    # gradient at layer 1
                    gradient_1 = np.dot(X_batch.T, error_1).T
                    # regularize except for bias
                    gradient_1[1:] -= regularization * self.w_1[1:]

                    # gradient at layer 2
                    gradient_2 = np.dot(z.T, error_2).T
                    # regularize except for bias
                    gradient_2[1:] -= regularization * self.w_2[1:]

                    self.v_1 = mc * self.v_1 - (1 - mc) * lr * gradient_1
                    self.v_2 = mc * self.v_2 - (1 - mc) * lr * gradient_2
                    self.w_1 += self.v_1
                    self.w_2 += self.v_2

                assert not np.any(np.isnan(self.w_1))
                assert not np.any(np.isnan(self.w_1))

                print "[w1l2] %5.4f" % np.linalg.norm(self.w_1)
                print "[w2l2] %5.4f" % np.linalg.norm(self.w_2)

                print "[g1l2] %5.4f" % np.linalg.norm(gradient_1)
                print "[g2l2] %5.4f" % np.linalg.norm(gradient_2)

                # calculate error rate of training data
                score_train = self.score(data_train, label_train)
                print "[train] %5.4f" % score_train
                scores_train.append(score_train)

                # calculate error rate of validation data
                score_valid = self.score(data_valid, label_valid)
                print "[valid] %5.4f" % score_valid
                scores_valid.append(score_valid)

                if score_valid > score_valid_best:
                    w_1_best = self.w_1
                    w_2_best = self.w_2
                    score_valid_best = score_valid
                    r_best = r+1

                measure_stop = time.clock()
                print measure_stop - measure_start

                print

        except KeyboardInterrupt:
            pass

        print "Best model: r = %d, correct rate = %f" % (
            r_best, score_valid_best)

        # show correct rate of train and valid
        plt.figure()
        plt.plot(np.arange(len(scores_train)), np.array(scores_train))
        plt.plot(np.arange(len(scores_valid)), np.array(scores_valid))
        plt.legend(['train', 'valid'])
        plt.show()

        self.w_1 = w_1_best
        self.w_2 = w_2_best

    def predict_proba(self, X):
        '''
        input ... X: feature vectors without bias
        outpu ... predicted label probabilities
        '''
        # add one dimention to future vector for bias
        n = len(X)                  # number of all data
        ones = np.ones((n, 1), dtype=np.float32)
        X = np.hstack((ones, X))

        a = np.dot(X, self.w_1.T)
        z = np.tanh(a)
        # add bias to z_pred_training
        ones = np.ones((len(X), 1), dtype=np.float32)
        z = np.hstack((ones, z))
        y = softmax(np.dot(z, self.w_2.T))
        return y

    def predict(self, X):
        '''
        input ... X: feature vectors without bias
        output ... predicted labels
        '''
        y = self.predict_proba(X)
        labels = np.argmax(y, axis=1)
        return labels

    def score(self, X, t):
        '''
        input ... X: feature vectors without corresponding point of bias
                  t: target labels
        output ... correct rate
        '''
        labels = self.predict(X)
        count_correct = np.sum(labels == t)
        return count_correct / float(len(X))


if __name__ == "__main__":

    # cross validation parameter, ratio of training and validation data
    n_train_rate = 0.9

    # load noisy sin data.
    x, t = generate_noisy_sin()
    plt.plot(x, t, '.')

    # cross validation
    data_train, data_valid, label_train, label_valid = \
        cross_validation.train_test_split(x, t,
                                          train_size=n_train_rate,
                                          random_state=0)

    classifier = NeuralNetworkClassifier(M=600)
    classifier.fit(data_train, label_train, data_valid, label_valid,
                   lr=0.0003, num_iteration=600, minibatch_size=500,
                   mc=0.9, regularization=0.001, std_w1_init=0.03,
                   std_w2_init=0.04)
