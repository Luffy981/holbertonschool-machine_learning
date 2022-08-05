#!/usr/bin/env python3
"""
Module contains function that uses
sklearn K-means algorithm.
"""


import sklearn.cluster as Cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset.

    Args:
        X: numpy.ndarray - (n, d) containing the dataset.
        k: Number of clusters.

    Return: C, clss
        C: numpy.ndarray - (k, d) Centroid means for
        each cluster.
        clss: numpy.ndarray - (n,) Index of the cluster in
        C that each data point belongs to.
    """

    k_means = Cluster.KMeans(k).fit(X)

    return k_means.cluster_centers_, k_means.labels_
