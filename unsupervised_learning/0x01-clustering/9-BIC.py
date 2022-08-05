#!/usr/bin/env python3
"""
Module contains function that finds the best
number of clusters for a Gaussian Mixture Model
using the Bayesian Information criterion.
"""


import numpy as np


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a Gaussian Mixture Model
    using the Bayesian Information criterion.

    Args:
        X: numpy.ndarray - (n, d) Data set.
        kmin: Positive integer, Minimum number of clusters to check for.
        kmax: Positive integer, Maximum number of clusters to check for.
            If kmax is None, kmax == number of points.
        iterations: Positive integer, maximum number of iterations
        for the algorithm.
        tol: Non-negative float containing tolerance of the log likelihood,
        used to determine early stopping i.e. if the difference is less
        than or equal to tol you should stop the algorithm.
        verbose: Boolean that determines if function should print information
        about the algorithm.

    Return: best_k, best_result, l, b, or None, None, None, None on failure.
        best_k: Best value for k based on its BIC.
        best_result: Tuple containing pi, m, S.
            pi: numpy.ndarray - (k,) Cluster priors for the best number
            of clusters.
            m: numpy.ndarray - (k, d) Centroid means for the best number
            of clusters.
            S: numpy.ndarray - (k, d, d) Covariance matrices for the best
            number of clusters.
        l: numpy.ndarray - (kmax - kmin + 1) Log likelihood for each cluster
        size tested.
        b: numpy.ndarray - (kmax - kmin + 1) Bayesian Information Criterion
        value for each cluster size tested.
    """

    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None, None, None
    if type(kmin) is not int or kmin <= 0:
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmax) is not int or kmax <= 0:
        return None, None, None, None
    if kmax <= kmin:
        return None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None

    # expectation_maximization = __import__('8-EM').expectation_maximization
    EM = expectation_maximization

    Ps, Ls = [], []
    n, d = X.shape

    for k in range(kmin, kmax + 1):

        Ps.append(1+(k*d)+(k*d))

        Ls.append((EM(X, k, iterations, tol, verbose))[-1])

    BICs = np.array(Ps)*np.log(n)-2*np.array(Ls)
    best_k = int(np.argmin(BICs) + kmin)
    priors, means, sigmas, _, _ = EM(X, best_k, iterations, tol, verbose)

    idxs = np.argsort(priors)[::-1]
    priors = priors[idxs]
    means = means[idxs]
    sigmas = sigmas[idxs, :]

    best_results = (priors, means, sigmas)

    return best_k, best_results, np.array(Ls), BICs
