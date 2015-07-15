# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 00:03:47 2015

@author: shimba
"""

import numpy as np
from utils import basic_distributions
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
    distributions = [basic_distributions.Gaussian(),  # 0
                     basic_distributions.Poisson(),   # 1
                     basic_distributions.Gamma(),     # 2
                     basic_distributions.Beta()       # 3
                     ]
    dist_type = 1
    sampler = distributions[dist_type]

    x = sampler()

    sampler.visualize(x)
    print "Distribution: ", sampler.get_name()

    average = x.sum().astype(np.float32) / len(x)
    my_poisson(average.round())
