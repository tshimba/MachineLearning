# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 15:49:47 2015

@author: shimba
"""

from utils import dataset

if __name__ == '__main__':
    dataset_dir = 'datasets'
    downloader = dataset.DownloadDataset(dataset_dir)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/"
    extension = ".txt"
    dataset_name = 'bag_of_words'
    downloader.bag_of_words(url, extension, dataset_name)
