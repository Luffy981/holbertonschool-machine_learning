#!/usr/bin/env python3
"""
Module contains init funtion for
T-SNE algorithm.
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


def softmax(X, diag_zero=True):
    """Take softmax of each row of matrix X."""

    e_x = np.exp(X)
    ex_sum = e_x.sum()-(np.diagonal(e_x)).sum()

    P = e_x / ex_sum

    if diag_zero:
        np.fill_diagonal(P, 1e-8)

    return P


def calc_prob_matrix(distances, sigmas=None):
    """
    Convert a distances matrix to a matrix of probabilities.
    """
    if sigmas is not None:
        two_sig_sq = 2. * np.square(
            sigmas.reshape((-1, 1))
            )
        return softmax(-distances / two_sig_sq)
    else:
        return softmax(distances)


def binary_search(eval_fn, target, tol=1e-10, max_iter=10000,
                  lower=1e-20, upper=1000.):
    """
    Perform a binary search over input values to eval_fn.

    Args:
        eval_fn: Function to optimise over.
        target: Target value.
        tol: Float, distance threshold to target.
        max_iter: Iterations to search for.
        lower: Lower bound of search range.
        upper: Upper bound of search range.
    Return:
        Best input value to function found during search.
    """
    for i in range(max_iter):
        guess = (lower + upper) / 2.
        val, entropy = eval_fn(guess)
        if val > target:
            upper = guess
        else:
            lower = guess
        if np.abs(val - target) <= tol:
            break
    return guess, entropy


def calc_perplexity(prob_matrix):
    """
    Calculate the perplexity of each row
    of a matrix of probabilities.
    """

    entropy = -np.sum(
        prob_matrix * np.log2(prob_matrix),
        1, dtype="float32"
        )
    perplexity = 2 ** entropy
    return perplexity, entropy[0]


def perplexity(distances, sigmas):
    """
    Wrapper function for quick calculation of
    perplexity over a distance matrix.
    """

    return calc_perplexity(calc_prob_matrix(distances, sigmas))


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a data set.
    Args:
        X: numpy.ndarray - (n, d) containing the dataset.
            n: number of data points.
            d: number of dimensions in each point.
        perplexity: perplexity that all Gaussian distributions should have.
        tol: maximum tolerance allowed for the difference in
          Shannon entropy from perplexity for all Gaussian distributions.

    Return:
        P affinities.
    """

    D, _, _, _ = P_init(X, perplexity)
    sigmas = find_optimal_sigmas(D, perplexity, tol)
    P = calc_prob_matrix(D, sigmas)
    np.fill_diagonal(P, 0.)
    return P


def find_optimal_sigmas(distances, target_perplexity, tol):
    """
    For each row of distances matrix, finds sigma that results
    in target perplexity for that role.
    """

    sigmas = []

    for i in range(distances.shape[0]):
        # fn that returns perplexity of this row given sigma
        eval_fn = lambda sigma: \
            perplexity(distances[i:i+1, :], np.array(sigma))
        correct_sigma, entropy = binary_search(
            eval_fn, target_perplexity, tol
            )
        sigmas.append(correct_sigma)
    return np.array(sigmas)
