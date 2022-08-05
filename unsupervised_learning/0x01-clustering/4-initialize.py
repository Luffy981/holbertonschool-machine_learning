#!/usr/bin/env python3
"""
Module contains initialization function
for gaussian mixture model.
"""


import numpy as np


def initialize(X, k):
    """
    Initializes variables for gaussian mixture model.

    Args:
        X: numpy.ndarray - (n, d) containing the data set.
        k: int, Number of clusters.

    Return: (pi, m, S) or (None, None, None)
        pi: numpy.ndarray - (k,) Priors for each cluster,
        initialized evenly.
        m: numpy.ndarray - (k, d) Centroid means for each cluster,
        initialized with K-means.
        S: numpy.ndarray - (k, d, d) Covariance matrices for each cluster,
        initialized as identity matrices.
    """

    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None

    # kmeans = __import__('1-kmeans').kmeans
    d = X.shape[1]

    m, _ = kmeans(X, k)
    S = np.repeat(np.identity(d)[np.newaxis, ...], k, axis=0)
    pi = np.full((k,), 1/k)

    return pi, m, S
