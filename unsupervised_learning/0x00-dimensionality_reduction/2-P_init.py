#!/usr/bin/env python3
"""
Initialize t-SNE
"""
import numpy as np


def squared_euc_dist(X):
    """
    Compute matrix containing negative squared euclidean
        distance for all pairs of points.
    Args:
        X: dataset (NxD)
    Return:
        D: D_ij = negative squared euclidean distance
            between rows X_i and X_j.
    """

    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)

    np.fill_diagonal(D, 0.)

    return D


def P_init(X, perplexity):
    """
    Init method for T-SNE algorithm.
    Args:
    X: numpy.ndarray (n, d) - contains dataset to be
        transformed by t-SNE
        n: number of data points.
        d: number of dimensions in each point.
    perplexity: perplexity that all Gaussian
        distributions should have.
    Return:
    D: a numpy.ndarray of shape (n, n) that calculates
        the squared pairwise distance between two data points.
    P: a numpy.ndarray of shape (n, n)
        initialized to all 0‘s that will contain the P affinities.
    betas: a numpy.ndarray of shape (n, 1)
        initialized to all 1’s that will contain all of the beta values.
    H is the Shannon entropy for perplexity perplexity with a base of 2.
    """

    n, d = X.shape

    betas, P = np.ones((n, 1)), np.zeros((n, n))
    D = squared_euc_dist(X)
    H = 1 / (np.log(2)/np.log(perplexity))

    return D, P, betas, H
