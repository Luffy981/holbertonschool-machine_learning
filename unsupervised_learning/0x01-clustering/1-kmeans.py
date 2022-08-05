#!/usr/bin/env python3
"""
Module contains kmeans function
that performs the K-means algorithm
on a dataset.
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


def Kassignments(X, clusters, assignments):
    """
    Returns list of indexes corresponding to
    min euclidean distance of each point in X to
    each point in clusters.
    """
    points, dims = X.shape
    x = X.reshape(points, 1, dims)
    x = x - np.repeat(clusters[np.newaxis, ...], points, axis=0)
    dist = np.linalg.norm(x, axis=2)

    mins = np.argmin(dist, axis=1)
    if np.array_equal(mins, assignments):
        return True, assignments

    return False, mins


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.

    Args:
        X: numpy.ndarray - (n, d) containing the dataset.
            n: Number of data points.
            d: Number of dimensions for each data point.
        k: Positive integer containing the number of clusters.
        iterations: Positive integer containing the maximum
          number of iterations that should be performed.

    Return: C, clss, or None, None on failure
        C: numpy.ndarray - (k, d) containing the centroid
        means for each cluster.
        clss: numpy.ndarray - (n,) containing the index of
          the cluster in C that each data point belongs to.

    """

    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    if type(k) is not int or k <= 0:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    clusters = initialize(X, k)
    assignments = np.zeros((X.shape[0],))

    if k == 1:
        return np.mean(X, axis=0)[np.newaxis, ...], assignments

    for i in range(iterations):
        done, assignments = Kassignments(X, clusters, assignments)

        if done:
            return clusters, assignments

        for j in range(k):
            idx = np.argwhere(assignments == j)
            if len(idx) == 0:
                clusters[j] = (initialize(X, 1))[0]
            else:
                clusters[j] = np.mean(X[idx, ...], axis=0)

    _, assignments = Kassignments(X, clusters, assignments)

    return clusters, assignments
