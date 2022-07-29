#!/usr/bin/env python3
"""
likelihood
"""
import numpy as np


def combination(n, k):
    """
       N choose K function.
       Expressed as:
         n!/k!(n-k)!
    """
    fact = np.math.factorial
    denominator = fact(k) * (fact(n-k))
    return fact(n)/denominator


def binomial_pdf(n, k, P):
    """
       Binomial probability density function.
       Args:
        n: Number of total samples.
        k: Number of positive cases.
        P: Positive case probability.
       Returns:
        Probability of k positive samples given
        n and P.
    """
    comb = combination(n, k)
    return comb*(P**k)*((1-P)**(n-k))


def likelihood(x, n, P):
    """
       Calculates the likelihood of obtaining data (x)
         in total samples (n) given various hypothetical
         probabilities (P) of developing severe side effects.
       Args:
        x: Number of patients that develop severe side effects.
        n: Total number of patients observed.
        P: 1D numpy.ndarray - Contains various hypothetical
         probabilities of developing severe side effects.
       Returns:
         1D numpy.ndarray - containing likelihood of obtaining
         the data, x and n, for each probability in P, respectively.
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        e = "x must be an integer that is greater than or equal to 0"
        raise ValueError(e)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if True in ((P < 0) + (P > 1)):
        e = "All values in P must be in the range [0, 1]"
        raise ValueError(e)

    likelihoods = np.zeros((P.shape[0],))

    for i in range(P.shape[0]):
        likelihoods[i] = binomial_pdf(n, x, P[i])

    return likelihoods
