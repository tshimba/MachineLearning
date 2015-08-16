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
    distributions = [basic_distributions.Gaussian(),        # 0
                     basic_distributions.Poisson(),         # 1
                     basic_distributions.Gamma(),           # 2
                     basic_distributions.Beta(),            # 3
                     basic_distributions.LinearModel(),     # 4
                     basic_distributions.LinearModel2(),    # 5
                     basic_distributions.PoissonGLM(),      # 6
                     ]

    # Data sampling
    dist_type = 0
    sampler = distributions[dist_type]
    x = sampler()

    N = len(x)
    mean = np.sum(x) / N
    std = np.sqrt(np.sum((x - mean) ** 2) / N)

    predicted_sampler = basic_distributions.Gaussian(mean=mean, std=std)
    px = predicted_sampler(num_examples=N)
    sampler.visualize(px)
