# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 00:03:47 2015

@author: shimba
"""

from utils import basic_distributions

if __name__ == '__main__':
    distributions = [basic_distributions.Gaussian(),  # 0
                     basic_distributions.Poisson(),   # 1
                     basic_distributions.Gamma(),     # 2
                     basic_distributions.Beta()       # 3
                     ]
    dist_type = 3
    sampler = distributions[dist_type]

    x = sampler()

    sampler.visualize(x)
    print "Distribution: ", sampler.get_name()