# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 00:03:47 2015

@author: shimba
"""

import numpy as np
from utils import mixture_distributions as mix
import matplotlib.pyplot as plt


def my_poisson(lam):
    n_s = 100000
    s = np.random.poisson(lam, n_s)
    plt.plot(range(n_s), s, '.')
    plt.show()

    plt.hist(s, 14, normed=True)
    plt.xlim([-2, 10])
    plt.show()

if __name__ == '__main__':
    distributions = [mix.MixtureOfGaussians(),  # 0
                     mix.MixtureOfPoissons(),   # 1
                     ]
    dist_type = 0
    sampler = distributions[dist_type]

    x = sampler(10000, complete_data=True)

    sampler.visualize(x[1])
    print "Distribution: ", sampler.get_name()
    print "Parameters: ", sampler.get_params()
