# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 17:49:55 2015

@author: shimba
"""

import requests
import os
from BeautifulSoup import BeautifulSoup


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
                with open(file_path, 'w') as f:
                    f.write(r.content)

if __name__ == '__main__':
    dataset_dir = '../datasets'
    downloader = DownloadDataset(dataset_dir)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/"
    extension = ".txt"
    dataset_name = 'bag_of_words'

    downloader.bag_of_words(url, extension, dataset_name)
