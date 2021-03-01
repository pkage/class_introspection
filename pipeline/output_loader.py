#! /usr/bin/env python3

import pickle
import os

import numpy as np
from pipeline import cluster_explanations


def load_explanations(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f'file not found: {filename}')

    with open(filename, 'rb') as pickle_file:
        return pickle.load(pickle_file)


def get_variances_labels(run, epsilon):
    out = {}
    for label in np.unique(run['y_tst_hw']):
        label = int(label)
        variances, cluster_lbls = cluster_explanations(run['shaps'], run['y_tst_hw'] == label, label, epsilon)
        out[label] = {
            'variances':    variances,
            'cluster_lbls': cluster_lbls
        }
    return out

def lbl_hist(data):
    out = {}
    for i in np.unique(data):
        out[int(i)] = np.count_nonzero(data == i)

    return out
        
