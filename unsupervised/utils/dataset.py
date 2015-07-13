# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 17:49:55 2015

@author: shimba
"""

import numpy as np
import requests
import os
from BeautifulSoup import BeautifulSoup
import gzip
from StringIO import StringIO


class DownloadDataset(object):
    def __init__(self, dataset_dir='datasets'):
        self.dataset_dir = dataset_dir
        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)
            print 'Directory "' + self.dataset_dir + '" is created.'

    def bag_of_words(self, url, extension, dataset_name, check_eachfile=False):
        dataset_path = self.dataset_dir + '/' + dataset_name
        # Exists directory and don't check each file
        if os.path.exists(dataset_path) and not check_eachfile:
            print 'This dataset already exists.'
            return
        # Don't exsists directory
        elif not os.path.exists(dataset_path):
            os.mkdir(dataset_path)
            print 'Directory "' + dataset_name + '" is created.'

        download_urls = []
        r = requests.get(url)
        soup = BeautifulSoup(r.content)
        links = soup.findAll('a')

        # extract urls
        for link in links:

            href = link.get('href')

            if href and extension in href:
                download_urls.append(href)

        for download_url in download_urls:

            file_name = download_url.split("/")[-1]
            file_path = dataset_path + '/' + file_name
            if os.path.exists(file_path):
                print '[' + file_name + '] already exists.'
                continue
            else:
                # absolute path link
                if url in download_url:
                    r = requests.get(download_url)
                # relative path link
                else:
                    r = requests.get(url + download_url)

                print 'Now saving the file [' + file_name + ']'
                if file_name.endswith('.gz'):
                    with open(file_path, 'wb') as f:
                        f.write(r.content)
                elif file_name.endswith('.txt'):
                    with open(file_path, 'w') as f:
                        f.write(r.content)


class LoadData(object):
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.file_list = os.listdir(dir_path)

    def load_data(self, data_name=None):
        '''
        DataNames:
            enron,
            kos,
            nips,
            nytimes,
            pubmed
        '''
        for file_name in self.file_list:
            if not file_name.endswith(".gz"):
                continue
            if data_name is not None and data_name not in file_name:
                continue
            file_name = os.path.join(self.dir_path, file_name)
            print 'opening', file_name
            with gzip.open(file_name, 'rb') as f:
                file_content = f.read()
            lines = file_content.split('\n')
            D = int(lines[0])
            W = int(lines[1])
            NNZ = int(lines[2])
            file_content = "\n".join(lines[3:])
            s = StringIO(file_content)
            data = np.loadtxt(s, dtype=np.int)
            break

        assert len(data) == NNZ, 'NNZ is incorrect'
        return D, W, NNZ, data

    def load_allocation_table(self, corpus_name):
        '''
        DataNames:
            enron,
            kos,
            nips,
            nytimes,
            pubmed
        '''
        for file_name in self.file_list:
            if not file_name.endswith(".txt"):
                continue
            if corpus_name is not None and corpus_name not in file_name:
                continue
            file_name = os.path.join(self.dir_path, file_name)

            with open(file_name, "r") as f:
                table = f.read()
            table = table.split('\n')
            if table[-1] == "":
                table = table[:-1]
            break
        return table

if __name__ == '__main__':
    dataset_dir = '../datasets'
    downloader = DownloadDataset(dataset_dir)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/"
    extension = ".txt"
    dataset_name = 'bag_of_words'

    downloader.bag_of_words(url, extension, dataset_name)
