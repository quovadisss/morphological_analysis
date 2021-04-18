import re
import time
import pickle
import math
import gensim
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


# get 300 dimensions w2v for each list
def get_w2v(x):
    new = []
    get_w2v_list = []
    for i in x:
        try:
            w2v = model[i]
            get_w2v_list.append(w2v)
            new.append(i)
        except KeyError:
            pass
    return get_w2v_list, new


# K means, word/Index dictionary
def k_means(vec, words, n_clusters):
    kmeans_clustering = KMeans(n_clusters=n_clusters, random_state=100)
    idx = list(kmeans_clustering.fit_predict(vec))
    word_centroid_map = {words[i]: idx[i] for i in range(len(words))}
    return word_centroid_map


def cluster_df(x, y):
    all_words = []
    for cluster in range(0, y):
        words = []
        for i in range(0, len(list(x.values()))):
            if(list(x.values())[i] == cluster):
                words.append(list(x.keys())[i])
        all_words.append(words)
    number = 0
    for i in all_words:
        if len(i) <= 5 and len(i) >= 3:
            number += 1
    return number, all_words


def onemore_clustering(words):

    def onemore(w, n):
        out = []
        for i in w:
            w2v, word = get_w2v(i)
            cluster = k_means(w2v, word, n)
            num, all_words = cluster_df(cluster, n)
            out.extend(all_words)
        return out

    result = []
    again15_30 = []
    again31_40 = []
    again40_50 = []
    for i in words:
        if len(i) > 15 and len(i) <= 30:
            again15_30.append(i)
        elif len(i) > 30 and len(i) <= 40:
            again31_40.append(i)
        elif len(i) > 40:
            again40_50.append(i)
        else:
            result.append(i)
    result.extend(onemore(again15_30, 4))
    result.extend(onemore(again31_40, 5))
    result.extend(onemore(again40_50, 6))

    return result


# Choose the best number of clusters
def best_cluster(x, name):
    vec, words = get_w2v(x)
    numbers = []
    n_clusters = []
    for i in range(int(len(vec)/12), int(len(vec)/3), 5):
        test_cluster = k_means(vec, words, i)
        number, all_words = cluster_df(test_cluster, i)
        numbers.append(number)
        n_clusters.append(i)
    n = n_clusters[numbers.index(max(numbers))]
    number, all_words = cluster_df(k_means(vec, words, n), n)
    result = onemore_clustering(all_words)
    result_df = pd.DataFrame(result)
    result_df.to_csv('data/{0}_{1}_df.csv'.format('fintech', name))


# -------------- Set parser to choose data set --------------
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='health_news',
                    help='Pick data to do clustering')
args = parser.parse_args()


# -------------- Get vectors for each attributes --------------
with open('data/morph_attr_{}.pkl'.format(args.data), 'rb') as fr:
    morpho = pickle.load(fr)

word2vec_loc = 'C:/GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(fname=word2vec_loc,
                                                        binary=True)

names = ['sys', 'tech', 'norp', 'org', 'prod']
for i, j in zip(morpho, names):
    best_cluster(i, j)
