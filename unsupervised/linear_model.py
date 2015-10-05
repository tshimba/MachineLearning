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
    dist_type = 4
    sampler = distributions[dist_type]
    y = sampler()

    # Calculate slope, intercept and error
    N = len(y)
    x = np.arange(0, N)
    A = np.vstack([x, np.ones(N)]).T

    lsm = np.linalg.lstsq(A, y)
    m = lsm[0][0]   # slope
    c = lsm[0][1]   # intercept
    # Sum of squared error to an error at one sample
    error = np.sqrt(lsm[1][0] / N)

    # Plot original sampling data
    sampler.visualize(y)
    print "Distribution: ", sampler.get_name()

    # Plot sampling data by using predicted m, c and error
    x_plot = np.arange(0, N)
    y_plot = (m * x_plot + c)
    noise = np.random.randn(N) * error
    y_plot = y_plot + noise
    sampler.visualize(y_plot)
