#!/usr/bin/env python3
"""
Module contains function for calculating
the maximization step in the EM algorithm
for a Gaussian Mixture Model.
"""


import numpy as np


def cov(X, P, mean):
    """
    Calculates the covariance of a dataset, given the mean.

    Args:
        X: numpy.ndarray - shape (n, d) containing data set.
            n: Number of data points.
            d: Number of dimensions in each data point.
        P: numpy.ndarray - (n, 1) Normalizer equivalent to
        the likelihood of rows X given a gaussian C with given mean,
        over n * the prior of C.
        belonging to gaussian with mean "mean".
        mean: numpy.ndarray - shape (d,) mean for calculation.

    Returns: cov
        covariance: numpy.ndarray - shape (d, d) containing the
            covariance matrix of the data set.
    """

    _, d = X.shape
    deviation = X - mean
    covariance = np.matmul(deviation.T, P*deviation)

    return covariance


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm
    for a Gaussian Mixture Model.

    Args:
        X: numpy.ndarray - (n, d) Data set.
        g: numpy.ndarray - (k, n) Posterior probabilities for
        each data point in each cluster.

    Return: priors, m, S, or None, None, None on failure
        priors: numpy.ndarray - (k,) Updated priors for each cluster.
        means: numpy.ndarray - (k, d) Updated centroid means for
        each cluster.
        S: numpy.ndarray - (k, d, d) Updated covariance matrices
        for each cluster.
    """

    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None, None
    if type(g) is not np.ndarray or g.ndim != 2:
        return None, None, None
    if g.shape[0] <= 0 or g.shape[1] != X.shape[0]:
        return None, None, None
    if False in np.isclose(g.sum(axis=0), np.ones((g.shape[1]))):
        return None, None, None

    k, n = g.shape

    priors = g.sum(axis=1)

    means = (g.reshape(k, n, 1)*X).sum(axis=1)
    means = means/priors[np.newaxis, ...].T

    # Covariance matrices
    S = []

    for i in range(k):
        S.append(cov(X, g[i][np.newaxis, ...].T/priors[i], means[i]))

    return priors/n, means, np.array(S)
