# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:36:34 2015

@author: shimba
"""

import numpy as np
from sklearn import datasets
from sklearn import decomposition

def f(a):
    if a >= 0:
        result = 1
    else:
        result = -1
    return result

#Load the digits dataset
digits = datasets.load_digits(2)    # load two classes, 0 and 1

# put rabels into variable y
y = [label for label in digits.target]
# zero to -1
for count in range(len(y)):
    if y[count] == 0:
        y[count] = -1
#print y

# 1 is the bias
tmp = [[1] for i in range(len(digits.images))]

# 2 dimention to 1 dimention -- this should be funciton
for count in range(len(digits.images)):
    for vec1 in digits.images[count]:
        for vec2 in vec1:
            tmp[count].append(vec2)
tmp = np.array(tmp)
pca = decomposition.PCA(3)
tmp_result = pca.fit_transform(tmp)
print tmp_result.shape

# vertical vector
xx = [0 for i in range(len(tmp_result))]
for count in range(len(tmp_result)):
    for vec1 in tmp_result[count]:
        xx[count] = np.matrix(tmp_result[count]).T
xx = np.array(xx)
print xx.shape

# initialize weight vector -- this should be function
w = np.array([0 for itr in range(3)])
w = np.matrix(w).T      # vertical vector

for itr in range(1):
    for count in range(len(xx)):
        ff = f(w.T * xx[count])
        if ff == y[count]:
            print True
        else:
            print False
            w = w - xx[count]*ff