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


def factorize(v, pc=10, iteration=50):
    ic = v.shape[0]
    fc = v.shape[1]

    w = np.random.random((ic, pc))
    h = np.random.random((pc, fc))

    for i in range(iteration):
        print 'epoch', i + 1
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


def showfeatures(w, h, table, out='features.txt'):
    for i in range(h.shape[0]):
        slist = []
        for j in range(h.shape[1]):
            slist.append((h[i, j], table[j]))
        slist.sort()
        slist.reverse()

        n = [s[1] for s in slist[0:6]]
        print n


def sparse_to_dense(D, W, NNZ, data):
    mat = np.zeros((D, W))
    for datum in data:
        d_id = datum[0] - 1
        w_id = datum[1] - 1
        n_w = datum[2]
        mat[d_id, w_id] = n_w
    return mat


if __name__ == '__main__':
    dataset_dir = 'datasets'
    downloader = dataset.DownloadDataset(dataset_dir)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/"
    extension = ".txt"
    dataset_name = 'bag_of_words'
    downloader.bag_of_words(url, extension, dataset_name)

    dir_path = os.path.join(dataset_dir, dataset_name)
    dataloder = dataset.LoadData(dir_path)
    data_name = 'kos'
    D, W, NNZ, data = dataloder.load_data(data_name)
    table = dataloder.load_allocation_table(data_name)

    mat = sparse_to_matrix(D, W, NNZ, data)

    w, h = factorize(mat, pc=16, iteration=50)
    showfeatures(w, h, table)
