# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:59:56 2015

@author: sakurai
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.datasets import fetch_mldata


def draw_filters(W, cols=20, fig_size=(10, 10), filter_shape=(28, 28),
                 filter_standardization=False):
    border = 2
    num_filters = len(W)
    rows = int(np.ceil(float(num_filters) / cols))
    filter_height, filter_width = filter_shape

    if filter_standardization:
        W = preprocessing.scale(W, axis=1)
    image_shape = (rows * filter_height + (border * rows),
                   cols * filter_width + (border * cols))
    low, high = W.min(), W.max()
    low = (3 * low + high) / 4
    high = (low + 3 * high) / 4
    all_filter_image = np.random.uniform(low=low, high=high,
                                         size=image_shape)
    all_filter_image = np.full(image_shape, W.min())

    for i, w in enumerate(W):
        start_row = (filter_height * (i / cols) +
                     (i / cols + 1) * border)
        end_row = start_row + filter_height
        start_col = (filter_width * (i % cols) +
                     (i % cols + 1) * border)
        end_col = start_col + filter_width
        all_filter_image[start_row:end_row, start_col:end_col] = \
            w.reshape(filter_shape)

    plt.figure(figsize=fig_size)
    plt.imshow(all_filter_image, cmap=plt.cm.gray,
               interpolation='none')
    plt.tick_params(axis='both',  labelbottom='off',  labelleft='off')
    plt.show()



if __name__ == '__main__':
    mnist = fetch_mldata('MNIST original')
    X = mnist.data[:90].astype(np.float32) / 255.0
    draw_filters(X, 10)