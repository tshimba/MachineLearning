# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 23:08:22 2015

@author: ryuhei
"""

import numpy as np
import matplotlib.pyplot as plt


class ProbabilityDistribution(object):
    def get_name(self):
        return self.__class__.__name__

    def get_params(self):
        return self.__dict__

    def visualize(self, data):
        plt.figure()
        self._plot(data)
        plt.figure()
        self._hist(data)
        plt.show()

    def _plot(self, data):
        plt.plot(data, '.')

    def _hist(self, data):
        x_min, x_max = int(min(data)), int(max(data)+1)
        bins = np.arange(x_min, x_max, 0.2)
        plt.hist(data, bins=bins, range=(x_min, x_max))


class Gaussian(ProbabilityDistribution):
    def __init__(self, mean=50, std=2):
        self.mean = mean
        self.std = std

    def __call__(self, num_examples=10000):
        return np.random.normal(self.mean, self.std, num_examples)


class Poisson(ProbabilityDistribution):
    def __init__(self, mean=3):
        self.mean = mean

    def __call__(self, num_examples=1000):
        return np.random.poisson(self.mean, num_examples)

    def _hist(self, data):
        x_min, x_max = int(min(data)), int(max(data)+1)
        bins = np.arange(x_min, x_max) - 0.5
        plt.hist(data, bins=bins, range=(x_min, x_max))


class Gamma(ProbabilityDistribution):
    def __init__(self, shape=2, scale=3):
        self.shape = shape
        self.scale = scale

    def __call__(self, num_examples=10000):
        return np.random.gamma(self.shape, self.scale, num_examples)


class Beta(ProbabilityDistribution):
    def __init__(self, alpha=5, beta=1):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, num_examples=10000):
        return np.random.beta(self.alpha, self.beta, num_examples)

    def _hist(self, data):
        bins = np.linspace(0, 1, 100)
        plt.hist(data, bins=bins, range=(0, 1))


if __name__ == '__main__':
    distributions = [Gaussian(),  # 0
                     Poisson(),   # 1
                     Gamma(),     # 2
                     Beta()       # 3
                     ]
    dist_type = 1
    sampler = distributions[dist_type]

    x = sampler()

    sampler.visualize(x)
    print "Distribution: ", sampler.get_name()
    print "Parameters: ", sampler.get_params()
