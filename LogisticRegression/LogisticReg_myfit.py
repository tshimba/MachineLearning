# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 14:40:08 2015

@author: shimba

nonononnafa
"""
# print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn import linear_model
from sklearn import datasets
from sklearn import cross_validation
from sklearn import grid_search

class LogisticRegression(BaseEstimator, RegressorMixin):

    def __init__(self, penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None):
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.C = C
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state

    def phi(self, x, y):
        return np.array([x, y, 1])

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
        
    def fit(self, X, y):
        self.w = np.random.randn(3)  # initialize parameter
        eta = 0.1
        # the gradient of error function
        numFeature = X.shape[0]

        for n in xrange(50):
            list = range(numFeature)
            np.random.shuffle(list)        
        
            for i in list:
                t_n = y[i]
                x_i, y_i = X[i, :]
                feature = self.phi(x_i, y_i)
                #print u"%d, %d" % (i, t_n)
                predict = self.sigmoid(np.inner(self.w, feature))     
                
                self.w -= eta * (predict - t_n) * feature
                #print "%f, %f" % (w[0]/w[2], w[1]/w[2])
                #print self.w
                #gradient_E = np.dot(feature.T, Y-T)
            
        eta *= 0.9
        #print eta
        #gradient_E = np.dot(feature.T, Y-T)
        return self
        
     

        

        

def LogisticRegression_main():
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target
    X = X[y != 2]
    y = y[y != 2]
    
    # parameter?
    # cs = l1_min_c(X, y, loss='log') * np.logspace(0, 3)
    
    #clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    
    #clf.fit(X, y)

    regressor = LogisticRegression()
    regressor.fit(X, y)
    
    # plot result
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    #Z = regressor.predict(np.c_[xx.ravel(), yy.ravel()])
    #Z = Z.reshape(xx.shape)
    #plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    
    # Plot points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    
    w = regressor.w
    
    # 図を描くための準備
    seq = np.arange(1, 8, 0.01)
    xlist, ylist = np.meshgrid(seq, seq)
    zlist = [regressor.sigmoid(np.inner(w, regressor.phi(a, b))) for a, b in zip(xlist, ylist)]
    
    # 散布図と予測分布を描画
    plt.imshow(zlist, extent=[1,8,1,8], origin='lower', cmap=plt.cm.PiYG_r)
    plt.plot(X[y==1,0], X[y==1,1], 'o', color='red')
    plt.plot(X[y==0,0], X[y==0,1], 'o', color='blue')
    plt.show()

if __name__ == '__main__':
    LogisticRegression_main()