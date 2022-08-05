#!/usr/bin/env python3
"""
Module contains initialization function
for K-means.
"""


import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means.
    Args:
        X: numpy.ndarray - (n, d) containing the dataset that
          will be used for K-means clustering
            n: Number of data points.
            d: Number of dimensions for each data point.
        k: Positive integer containing the number of clusters.
    Return:
        numpy.ndarray - (k, d) containing the initialized centroids
          for each cluster, or None on failure.
    """
    try:
        if k <= 0:
            return None
        _, d = X.shape

        low = np.amin(X, axis=0)
        high = np.amax(X, axis=0)

        return np.random.uniform(low, high, (k, d))
    except Exception:
        return None
