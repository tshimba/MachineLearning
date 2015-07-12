# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 15:49:47 2015

@author: shimba
"""

import numpy as np
from utils import dataset
import os


def difcost(a, b):
    dif = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            dif += pow(a[i, j] - b[i, j], 2)
    return dif


def factorize(v, pc=10, iter=50):
    ic = v.shape[0]
    fc = v.shape[1]

    w = np.random.random((ic, pc))
    h = np.random.random((pc, fc))

    for i in range(iter):
        wh = np.dot(w, h)

        cost = difcost(v, wh)

        if i & 10 == 0:
            print '[cost]', cost

        if cost == 0:
            break

        hn = np.dot(w.T, v)
        hd = np.dot(np.dot(w.T, w), h)

        h = h * hn / hd

        wn = np.dot(v, h.T)
        wd = np.dot(np.dot(w, h), h.T)

        w = w * wn / wd

    return w, h

if __name__ == '__main__':
    dataset_dir = 'datasets'
    downloader = dataset.DownloadDataset(dataset_dir)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/"
    extension = ".txt"
    dataset_name = 'bag_of_words'
    downloader.bag_of_words(url, extension, dataset_name)

    m1 = np.array(([1, 2, 3], [4, 5, 6]))
    m2 = np.array(([1, 2], [3, 4], [5, 6]))
    w, h = factorize(np.dot(m1, m2), pc=3, iter=100)
    print w.shape
    print h.shape
    print np.dot(w, h)

    dir_path = os.path.join(dataset_dir, dataset_name)
    dataloder = dataset.LoadData(dir_path)
    data_name = 'enron'
    D, W, NNZ, data = dataloder.load_data(data_name)
    table = dataloder.load_allocation_table(data_name)

    print D, W, NNZ, data.shape
    print len(table)

    Mat = np.empty((D, W))
