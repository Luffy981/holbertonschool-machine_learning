#!/usr/bin/env python3
"""
Module contains function that calculates the
expectation step in the EM algorithm for a GMM.
"""


import numpy as np


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM
    algorithm for a GMM.

    Args:
        X: numpy.ndarray - (n, d) Data set.
        pi: numpy.ndarray - (k,) Priors for each cluster.
        m: numpy.ndarray - (k, d) Centroid means for each cluster.
        S: numpy.ndarray - (k, d, d) Covariance matrices
        for each cluster.

    Return: g, l, or None, None on failure
        g: numpy.ndarray - (k, n) Posterior probabilities for each
        data point in each cluster.
        l: Total log likelihood.
    """
    # print(np.isclose(pi.sum(), 1))
    # exit()

    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    if type(m) is not np.ndarray or m.ndim != 2:
        return None, None
    if m.shape[1] != X.shape[1]:
        return None, None
    if type(S) is not np.ndarray or S.ndim != 3:
        return None, None
    if S.shape[2] != S.shape[1] or S.shape[1] != X.shape[1]:
        return None, None
    if type(pi) is not np.ndarray or pi.ndim != 1:
        return None, None
    if S.shape[0] != pi.size or False in [np.isclose(pi.sum(), 1)]:
        return None, None

    # pdf = __import__('5-pdf').pdf
    k, _ = m.shape
    pi = pi.reshape(k, 1)

    likelihoods = np.array([
        pdf(X, m[i], S[i]) for i in range(k)
    ])

    numerator = likelihoods*pi

    tot_log_lik = numerator.sum(axis=0)
    tot_log_lik = np.log(tot_log_lik).sum()

    norm = numerator.sum(axis=0)[np.newaxis, ...]

    return numerator/norm, tot_log_lik
