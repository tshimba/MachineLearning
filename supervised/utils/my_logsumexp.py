# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:57:11 2015

@author: dashu2326
"""

# Reference: https://github.com/scipy/scipy/blob/2d1bd60e0330bb07880f8eab717e21ee17422245/scipy/misc/common.py

import numpy as np

def my_logsumexp(a):
    # fetch max value from a
    # keep array format
    a_max = np.amax(a, keepdims=True)

    if not np.isfinite(a_max):  # if infinit -> a_max is 0
        a_max = 0

    # suppress warnings about log of zero: means ln(0).
    with np.errstate(divide='ignore'):
        out = np.log(sum(np.exp(a - a_max)))    # logsumexp

    # 1 dimension to scalar value
    a_max = np.squeeze(a_max)

    out += a_max

    return out

if __name__ == "__main__":
    print 'Usage: call from softmax-regression.py'