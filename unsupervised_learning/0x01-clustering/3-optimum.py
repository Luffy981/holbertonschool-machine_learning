#!/usr/bin/env python3
"""
Module contains function that
tests for the optimum number of
clusters by variance.
"""


import numpy as np


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance.

    Args:
        X: numpy.ndarray - (n, d) containing the data set.
        kmin: positive integer containing the
        minimum number of clusters to check for.
        kmax: positive integer containing the maximum
        number of clusters to check for.
        iterations: positive integer containing the maximum
        number of iterations for K-means.

    Return: results, d_vars, or None, None on failure
        results: list containing the outputs of K-means.
        d_vars: list containing the difference in variance
        from the smallest cluster size for each cluster size.
    """

    # kmeans = __import__('1-kmeans').kmeans
    # variance = __import__('2-variance').variance
    results, d_vars = [], []

    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    if type(kmin) is not int or kmin <= 0:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmax) is not int or kmax <= 0:
        return None, None
    if kmax <= kmin:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    for i in range(kmin, kmax+1):
        clusters, assignments = kmeans(X, i, iterations)

        if i == kmin:
            org_variance = variance(X, clusters)
            results.append((clusters, assignments))
            d_vars.append(0.0)
            continue

        variance_diff = abs(org_variance-variance(X, clusters))

        results.append((clusters, assignments))
        d_vars.append(variance_diff)

    return results, d_vars
