# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:18:59 2015

@author: shimba
"""

from setuptools import setup, find_packages
import sys
 
sys.path.append('./unsupervised')
sys.path.append('./tests')

setup(
    name = 'Unsupervised',
    version = '0.1',
    description='Unsupervised Machine Learning Practice',
    packages = find_packages(),
    test_suite = 'test_complete_mixture_model.suite'
)
