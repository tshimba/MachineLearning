# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:36:34 2015

@author: shimba
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# copy from http://d.hatena.ne.jp/pashango_p/20090617/1245221088
'''
def PCA(P):
    m = sum(P) / float(len(P))
    P_m = P - m
    l,v = numpy.linalg.eig( numpy.dot(P_m.T,P_m) )
    return v.T
'''

def f(a):
    if a >= 0.:
        result = 1
    else:
        result = -1
    return result

# another reference: http://labs.cybozu.co.jp/blog/nakatani/2009/04/perceptron_1.html

#Load the digits dataset
digits = datasets.load_digits(2)    # load two classes, 0 and 1
#print digits.data.shape
#print len(digits.images)    # 360 samples

#Display the first digit
'''
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
'''

# create histgram

# create feature vector
x = [feature[:] for feature in digits.images]
#print x[0]

# put rabels into variable y
y = [label for label in digits.target]
# zero to -1
for count in range(len(y)):
    if y[count] == 0:
        y[count] = -1
#print y

# TODO
# visualize histgram of digits and select good feature vector

# initialize weight vector -- this should be function
#w = np.array([rand for rand in np.random.rand(65)])
w = np.array([0 for itr in range(65)])
# vertical vector
w = np.matrix(w).T
print w.shape
#print w
#print w * x[0]

# 1 is the bias
tmp = [[1] for i in range(len(digits.images))]
xx = [0 for i in range(len(digits.images))]

# 2 dimention to 1 dimention -- this should be funciton
for count in range(len(digits.images)):
    #fvec[count].append(1)
    for vec1 in digits.images[count]:
        for vec2 in vec1:
            tmp[count].append(vec2)
        xx[count] = np.matrix(tmp[count]).T

xx = np.array(xx)
print xx.shape

for itr in range(2000):
    for count in range(len(xx)):
        ff = f(w.T * xx[count])
        if ff == y[count]:
            print True
        else:
            print False
            w = w + xx[count]*ff