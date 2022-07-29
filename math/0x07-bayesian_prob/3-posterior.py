#!/usr/bin/env python3
"""
   Module contains likelihood, intersection,
   marginal, and posterior functions.
"""


from typing import Type
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


def Intersection(likelihoods, Pr):
    """
       Calculates the intersection of obtaining this
         data with the various hypothetical probabilities.

       Args:
        likelihoods: numpy.ndarray of likelihoods
        Pr: 1D numpy.ndarray - containing the prior beliefs of P.

       Return:
         1D numpy.ndarray - containing the intersection of obtaining
           x and n with each probability in P, respectively.
    """

    return likelihoods*Pr


def marginal(intersection):
    """
       Calculates the marginal probability
         of obtaining the data.

       Args:
        intersection: numpy.ndarray

       Return:
        The marginal probability of obtaining x and n.
    """

    return intersection.sum()


def posterior(x, n, P, Pr):
    """
       Calculates the posterior probability for the various
         hypothetical probabilities of developing severe side effects
         given the data, using bayes rule.

       Args:
        x: Number of patients that develop severe side effects.
        n: Total number of patients observed.
        P: 1D numpy.ndarray - containing the various
          hypothetical probabilities of developing severe side effects
        Pr: 1D numpy.ndarray - containing the prior beliefs of P.

       Return:
        Posterior probability of each probability in P given x and n.
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

    type_chk = type(Pr) is not np.ndarray

    if type_chk or P.shape != Pr.shape:
        e = "Pr must be a numpy.ndarray with the same shape as P"
        raise TypeError(e)

    if True in ((P < 0) + (P > 1)):
        e = "All values in P must be in the range [0, 1]"
        raise ValueError(e)
    if True in ((Pr < 0) + (Pr > 1)):
        e = "All values in Pr must be in the range [0, 1]"
        raise ValueError(e)

    if False in np.isclose([Pr.sum()], [1]):
        raise ValueError("Pr must sum to 1")

    likelihoods = likelihood(x, n, P)
    marginal_prob = marginal(Intersection(likelihoods, Pr))

    return (likelihoods*Pr)/marginal_prob
