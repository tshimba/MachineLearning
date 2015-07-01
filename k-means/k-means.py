# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 02:32:57 2015

@author: shimba
"""

import numpy as np
import matplotlib.pyplot as plt
import generate_clustering_data
import itertools


class KMeans(object):
    def __init__(self, K=2):
        self.K = K

    def fit(self, X, n_epoch):
        centroids = self.init_centroids(X)

        try:
            for epoch in xrange(0, n_epoch):
                labels = self.e_step(X, centroids)
                print
                print 'Iteration', epoch + 1, ': ', '=== E Step ==='
                self.preview_stage(X, centroids, labels)

                self.m_step(X, centroids, labels)
                print 'Iteration', epoch + 1, ': ', '=== M Step ==='
                self.preview_stage(X, centroids, labels)
                self.score(X, centroids, labels)

        except KeyboardInterrupt:
            pass

        self.centroids = centroids

    def transform(self, X):
        pass

    def e_step(self, X, centroids):
        # update label
        labels = self.nearest_neighbor(X, centroids)
        return labels

    def m_step(self, X, centroids, labels):
        for k in range(self.K):
            points = X[labels == k]
            centroids[k] = points.mean(axis=0)
        return centroids

    def preview_stage(self, X, centroids, labels):
        n = X.shape[-1]
        if n <= 4:
            plt.figure(figsize=(10, 6))
            for i, (x, y) in enumerate(itertools.combinations(range(n), 2)):
                plt.subplot(2, 3, i + 1)
                for k in range(self.K):
                    color = plt.cm.get_cmap('nipy_spectral')(k * 255 / self.K)
                    plt.scatter(
                        X[labels == k, x],
                        X[labels == k, y],
                        marker='x',
                        c=color,
                    )
                    plt.scatter(
                        centroids[:, x],
                        centroids[:, y],
                        marker='*',
                        c='yellow',
                        s=200,
                    )
            plt.autoscale()
            plt.show()
        else:
            for k in range(self.K):
                plt.matshow(centroids[k].reshape(8, 8), cmap=plt.cm.gray)
                plt.show()
                plt.draw()

    def score(self, X, centroids, labels):
        distances = np.sum((np.expand_dims(X, 1) - centroids) ** 2, axis=2)
        score = np.sum(distances[range(len(X)), labels])
        print '[score]', score

    def init_centroids(self, X):
        indices = np.random.choice(len(X), self.K, False)
        centroids = X[indices]
        return centroids

    def nearest_neighbor(self, X, centroids):
        distances = np.sum((np.expand_dims(X, 1) - centroids) ** 2, axis=2)
        labels = distances.argmin(axis=1)
        return labels


if (__name__ == '__main__'):
    np.random.seed(0)

    K = 4
    n_epoch = 5

    X = generate_clustering_data.load_data(data_name='iris')

    kmeans = KMeans(K=K)
    kmeans.fit(X, n_epoch)
