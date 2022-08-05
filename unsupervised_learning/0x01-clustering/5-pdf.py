#!/usr/bin/env python3
"""
Module contains function that
calculates the probability density
function for a gaussian distribution.
"""


import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function
    of a Gaussian distribution.

    Args:
        X: numpy.ndarray - (n, d) Data points whose
        PDF should be evaluated.
        m: numpy.ndarray - (d,) Mean of the distribution.
        S: numpy.ndarray - (d, d) Covariance of the
        distribution.

    Return: P, or None on failure
    P is a numpy.ndarray of shape (n,) PDF values for
    each data point.
    """

    if type(X) is not np.ndarray or X.ndim != 2:
        return None
    if type(m) is not np.ndarray or m.shape[0] != X.shape[1]:
        return None
    if type(S) is not np.ndarray or S.ndim != 2:
        return None
    if S.shape[0] != S.shape[1] or S.shape[0] != X.shape[1]:
        return None

    D = m.shape[0]

    Px = (2*np.pi)**(D/2)
    Px = 1 / (Px * (np.linalg.det(S)**(1/2)))
    covI = np.linalg.inv(S)
    x_mu = X - m.reshape(D, 1).T
    dot = np.dot(x_mu, covI)
    print(covI)

    # dot where i == j
    dot = (dot * x_mu).sum(axis=1)

    pdv = Px*np.exp((-1/2)*dot)


    # Set minimum value
    pdv[pdv < 1e-300] = 1e-300

    return pdv
