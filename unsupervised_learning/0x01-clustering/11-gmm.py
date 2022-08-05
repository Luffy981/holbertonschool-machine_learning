#!/usr/bin/env python3
"""
Module contains function for performing Gaussian
Mixture Model algorithm using sklearn.
"""


import sklearn.mixture as Mixture


def gmm(X, k):
    """
    Calculates Gaussian Mixture Models from a dataset.

    Args:
        X: numpy.ndarray - (n, d) Dataset.
        k: Number of clusters.

    Return: priors, m, S, labels, bic
        priors: numpy.ndarray - (k,) Cluster priors.
        m: numpy.ndarray - (k, d) Centroid means.
        S: numpy.ndarray - (k, d, d) Covariance matrices.
        labels: numpy.ndarray - (n,) Cluster indices for each
        data point.
        bic: numpy.ndarray - (kmax - kmin + 1) BIC value for each cluster
        size tested.
    """

    mix = Mixture.GaussianMixture(k).fit(X)

    priors = mix.weights_
    means = mix.means_
    sigmas = mix.covariances_
    labels = mix.predict(X)
    bic = mix.bic(X)

    return priors, means, sigmas, labels, bic
