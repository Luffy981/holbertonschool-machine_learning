#!/usr/bin/env python3
"""
Module contains functions for executing
expectation maximization algorithm for
gaussian mixture model.
"""


import numpy as np


def expectation_maximization(
        X, k, iterations=1000,
        tol=1e-5, verbose=False
):
    """
    Executes expectation maximization algorithm for
    gaussian mixture model.

    Args:
        X: numpy.ndarray - (n, d) Data set.
        k: Positive integer, number of clusters.
        iterations: Positive integer, maximum number of iterations
        for the algorithm.
        tol: Non-negative float containing tolerance of the log likelihood,
        used to determine early stopping i.e. if the difference is less
        than or equal to tol you should stop the algorithm.
        verbose: Boolean that determines if function should print information
        about the algorithm.
            If True, print "Log Likelihood after {i} iterations: {l}"
            every 10 iterations and after the last iteration, where
            {i} is the number of iterations of the EM algorithm and
            {l} is the log likelihood, rounded to 5 decimal places.

    Return: pi, m, S, g, l, or None, None, None, None, None

    """

    if type(X) is not np.ndarray or X.ndim != 2:
        print("1")
        return None, None, None, None, None
    if type(k) is not int or k <= 0:
        print("2 ERROR HERE")
        return None, None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None, None
    if type(verbose) is not bool:
        print("last")
        return None, None, None, None, None

    # initialize = __import__('4-initialize').initialize
    # expectation = __import__('6-expectation').expectation
    # maximization = __import__('7-maximization').maximization

    # Initialize priors, means, and covariance matrices
    # for representing k gaussian distributions, init using
    # the K-means algorithm.

    priors, means, sigmas = initialize(X, k)

    # Likelihoods from step t - 1 and current step.

    Lk = [0, 0]

    for i in range(iterations):

        posteriors, Lk[1] = expectation(
            X, priors, means, sigmas
        )

        if abs(Lk[0]-Lk[1]) <= tol:
            Verbose(i, Lk[1].round(5), verbose, end=True)
            return priors, means, sigmas, posteriors, Lk[1]

        priors, means, sigmas = maximization(X, posteriors)

        Verbose(i, Lk[1].round(5), verbose)

        Lk[0] = Lk[1]

    Verbose(iterations, Lk[1].round(5), verbose, end=True)

    return priors, means, sigmas, posteriors, Lk[1]


def Verbose(iter, Lk, verbose, end=False):
    """Verbose helper function for EM algorithm."""

    if verbose is False:
        return
    if iter % 10 == 0 or end:
        print("Log Likelihood after {} iterations: {}".format(iter, Lk))
