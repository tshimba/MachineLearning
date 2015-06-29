# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 02:32:57 2015

@author: shimba
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import loadtxt


def load_faithful():
    return loadtxt('faithful.txt')

class KMeansClassifier(object):
    def __init__(self, K=2):
        self.K = K


    def fit(self, X, n_epoch):
        cluster = self.init_cluster(X)
        labels = np.zeros((X.shape[0], cluster.shape[0]))    # 1 of K
        
        try:
            for epoch in xrange(0, n_epoch):
                labels = self.EStep(X, cluster)
                print '[E Step]'
                self.preview_stage(X, cluster, labels)
                self.MStep(X, cluster, labels)
                print '[M Step]'
                self.preview_stage(X, cluster, labels)
        except KeyboardInterrupt:
            pass


    def EStep(self, X, cluster):
        # update label
        labels = self.nearest_neighbor(X, cluster)
        return labels


    def MStep(self, X, cluster, labels):
        pass


    def preview_stage(self, X, cluster, labels):
        plt.figure()
        color_id = np.array(('red', 'blue', 'green', 'yellow', 'black'))
        for i, label in enumerate(labels):
            for j in range(labels.shape[-1]):
                if label.argmax() == j:
                    plt.plot(X[i, 0], X[i, 1], 'o', color=color_id[j])
        plt.plot(cluster[:, 0], cluster[:, 1], 'gx')
        plt.show()


    def init_cluster(self, X):
        cluster = np.random.uniform(size=(self.K, X.shape[-1]))
        # Adjust the scale
        for index in np.arange(X.shape[-1]):
            cluster[:, index] *= np.max(X[:, index])
        return cluster

    def distance(self, p1, p2):
        return np.sum((p1 - p2) ** 2)

    def nearest_neighbor(self, X, cluster):
        distance = np.array([self.distance(x, cluster[0]) for x in X])
        distances = distance.reshape(-1, 1)
        for c in cluster[1:]:
            distance = np.array([self.distance(x, c) for x in X])
            distance = distance.reshape(-1, 1)
            distances = np.hstack((distances, distance))
        nearest_points = distances.argmin(axis=1)
        # create 1 of K label
        labels = np.zeros((X.shape[0], cluster.shape[0]))
        for i in np.arange(labels.shape[0]):
            labels[i].put(nearest_points[i], 1)
        return labels


if (__name__ == '__main__'):
    np.random.seed(0)
    
    K = 2
    n_epoch = 1
    
    X = load_faithful()

    classifier = KMeansClassifier(K=K)
    classifier.fit(X, n_epoch)
