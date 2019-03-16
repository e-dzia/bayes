from scipy import cluster
import pandas as pd
import numpy as np


NO_DISCRETIZATION = 0
EQUAL_WIDTH = 1
EQUAL_FREQUENCY = 2
K_MEANS = 3


def discretize_equal_width(X, num_of_bins):
    bins = []
    for column in X.columns:
        bins = np.linspace(start=min(X[column]), stop=max(X[column]), num=num_of_bins)
        # print(self.bins)
        X.loc[:, column] = np.digitize(X[column], bins, right=True)
    # print(X)
    return X


def discretize_k_means(X, num_of_bins):
    bins = []
    for column in X.columns:
        bins, distortion = cluster.vq.kmeans(X[column].astype(float), num_of_bins)
        bins.tolist()
        bins.sort()
        X.loc[:, column] = np.digitize(X[column], bins, right=True)
    # print(X)
    return X


def discretize_equal_frequency(X, num_of_bins):
    bins = []
    for column in X.columns:
        column_values = X[column].values.copy()
        column_values.sort()
        splitted_values = np.array_split(column_values, num_of_bins)
        bins = []
        for array in splitted_values:
            bins.append(min(array))
        X.loc[:, column] = np.digitize(X[column], bins, right=True)
    return X