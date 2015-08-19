# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:20:43 2015

@author: shimba
"""

import unittest
from complete_mixture_model import CompleteMixtureModelParameterPredictor
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
        x = (np.array((0, 0, 1, 1)), np.array((-2, 2, 12, 18)))
        return x
        
    def get_params(self):
        return self.__dict__
    

class ParameterPredictorTest(unittest.TestCase):

    def setUp(self):
        self.gaussian_mixture_generator()

    def gaussian_mixture_generator(self):
        sampler = FixedMixtureOfGaussians()
        x = sampler()
        print x
        self.params = sampler.get_params()
        
        predictor = CompleteMixtureModelParameterPredictor()
        self.K, self.weights, self.means, self.stds = predictor.predict(x)
        
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
    suite.addTests(unittest.makeSuite(ParameterPredictorTest))
    return suite
