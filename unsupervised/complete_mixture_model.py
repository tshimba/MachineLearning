# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 00:03:47 2015

@author: shimba
"""

import numpy as np
from utils import mixture_distributions as mix

# get complete data

class CompleteMixtureModelParameterPredictor(object):
    def predict(self, x):
        N = x[0].size
        classes = np.unique(x[0])
        N_classes = classes.size
        sum_each_class = np.empty(N_classes)
        each_class_x = []
        for i, c in enumerate(classes):
            class_x = x[1][x[0] == c]
            each_class_x.append(class_x)
            sum_each_class[i] = class_x.size
    
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

    x = sampler(10000, complete_data=True)

    # Show sampled original data
    sampler.visualize(x[1])
    print "Distribution: ", sampler.get_name()
    print "Parameters: ", sampler.get_params()

    predictor = CompleteMixtureModelParameterPredictor()
    K, weights, means, stds = predictor.predict(x)

    predicted_sampler = mix.MixtureOfGaussians(K=K, weights=weights,
                                               means=means, stds=stds)
    px = predicted_sampler(num_examples=10000)

    # Show sampled data by using predicted parameters
    sampler.visualize(px)
