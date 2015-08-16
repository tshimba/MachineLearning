# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 00:03:47 2015

@author: shimba
"""

import numpy as np
from utils import mixture_distributions as mix

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

    N = len(x[0])
    classes = np.unique(x[0])
    N_classes = len(classes)
    sum_each_class = np.empty(N_classes)
    each_class_x = []
    for i, c in enumerate(classes):
        class_x = x[1][x[0] == c]
        each_class_x.append(class_x)
        sum_each_class[i] = len(class_x)

    # Compute weights
    weights = sum_each_class / float(N)

    # Compute means and stds
    means = np.empty(N_classes)
    stds = np.empty(N_classes)
    for i, c in enumerate(classes):
        means[i] = np.sum(each_class_x[i]) / sum_each_class[i]
        stds[i] = np.sqrt(np.sum((each_class_x[i] - means[i]) ** 2) /
                          sum_each_class[i])

    predicted_sampler = mix.MixtureOfGaussians(K=N_classes, weights=weights,
                                               means=means, stds=stds)
    px = predicted_sampler(num_examples=N)

    # Show sampled data by using predicted parameters
    sampler.visualize(px)
