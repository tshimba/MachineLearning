# -*- coding: utf-8 -*-
"""
Author: Olivier Grisel <olivier.grisel@ensta.org>
        Lars Buitinck <L.J.Buitinck@uva.nl>
License: BSD 3 clause
Reference: http://scikit-learn.org/stable/auto_examples/applications/
           topics_extraction_with_nmf.html
"""

import numpy as np
from sklearn.decomposition import NMF
import os
from utils import dataset


def sparse_to_matrix(D, W, NNZ, data):
    mat = np.zeros((D, W))
    for datum in data:
        d_id = datum[0] - 1
        w_id = datum[1] - 1
        n_w = datum[2]
        mat[d_id, w_id] = n_w
    return mat


def showfeatures(h, table, out='features.txt'):
    for i in range(h.shape[0]):
        slist = []
        for j in range(h.shape[1]):
            slist.append((h[i, j], table[j]))
        slist.sort()
        slist.reverse()

        n = [s[1] for s in slist[0:6]]
        print n

if __name__ == '__main__':
    n_topics = 16

    dataset_dir = 'datasets'
    dataset_name = 'bag_of_words'
    dir_path = os.path.join(dataset_dir, dataset_name)
    dataloder = dataset.LoadData(dir_path)
    data_name = 'kos'
    D, W, NNZ, data = dataloder.load_data(data_name)
    table = dataloder.load_allocation_table(data_name)
    tfidf = sparse_to_matrix(D, W, NNZ, data)

    # Fit the NMF model
    nmf = NMF(n_components=n_topics, random_state=1).fit(tfidf)
    showfeatures(nmf.components_, table)
