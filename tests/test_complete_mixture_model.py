# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:20:43 2015

@author: shimba
"""

import unittest
from complete_mixture_model import CompleteMixtureModelParameterEstimator
import numpy as np


class FixedMixtureOfGaussians(object):
    def __init__(self, K=2, weights=[0.5, 0.5],
                 means=[0, 15], stds=[2, 3]):
        assert K == len(means) == len(stds)
        self.K = K
        self.weights = weights
        self.means = means
        self.stds = stds

    def __call__(self):
        z = np.array((0, 0, 1, 1))
        x = np.array((-2, 2, 12, 18))
        return z, x

    def get_params(self):
        return self.__dict__


class ParameterEstimatorTest(unittest.TestCase):

    def setUp(self):
        self.gaussian_mixture_generator()

    def gaussian_mixture_generator(self):
        sampler = FixedMixtureOfGaussians()
        z, x = sampler()
        self.params = sampler.get_params()

        estimator = CompleteMixtureModelParameterEstimator()
        self.K, self.weights, self.means, self.stds = estimator.estimate(z, x)

    def test_K(self):
        assert(np.allclose(self.params['K'], self.K))

    def test_weights(self):
        assert(np.allclose(self.params['weights'], self.weights))

    def test_means(self):
        assert(np.allclose(self.params['means'], self.means))

    def test_stds(self):
        assert(np.allclose(self.params['stds'], self.stds))


def suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.makeSuite(ParameterEstimatorTest))
    return suite
