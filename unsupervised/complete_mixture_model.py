# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 00:03:47 2015

@author: shimba
"""

import numpy as np
from utils import mixture_distributions as mix

# get complete data

class CompleteMixtureModelParameterEstimator(object):
    def estimate(self, z, x):
        N = len(z)
        classes = np.unique(z)
        N_classes = len(classes)
        sum_each_class = np.empty(N_classes)
        each_class_x = []
        for i, c in enumerate(classes):
            class_x = x[z == c]
            each_class_x.append(class_x)
            sum_each_class[i] = len(class_x)

        # Compute weights
        weights = sum_each_class / float(N)

        # Compute means and stds
        means = np.empty(N_classes)
        stds = np.empty(N_classes)
        for i, c in enumerate(classes):
            means[i] = np.mean(each_class_x[i])
            stds[i] = np.std(each_class_x[i])
        return N_classes, weights, means, stds

if __name__ == '__main__':
    distributions = [mix.MixtureOfGaussians(),  # 0
                     mix.MixtureOfPoissons(),   # 1
                     ]
    dist_type = 0
    sampler = distributions[dist_type]

    z, x = sampler(10000, complete_data=True)

    # Show sampled original data
    sampler.visualize(x)
    print "Distribution: ", sampler.get_name()
    print "Parameters: ", sampler.get_params()

    estimator = CompleteMixtureModelParameterEstimator()
    K, weights, means, stds = estimator.estimate(z, x)

    estimated_sampler = mix.MixtureOfGaussians(K=K, weights=weights,
                                               means=means, stds=stds)
    px = estimated_sampler(num_examples=10000)

    # Show sampled data by using estimated parameters
    sampler.visualize(px)
