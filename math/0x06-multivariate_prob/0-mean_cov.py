#!/usr/bin/env python3
"""
   Module containing function for
   computing mean and covariance.
"""


import numpy as np


def mean_cov(X):
    """
       Calculates the mean and covariance of
        a dataset.

       Args:
        X: numpy.ndarray - shape (n, d) Containing data set.
            n: Number of data points.
            d: Number of dimensions in each data point.
       Returns: mean, cov:
        mean: numpy.ndarray - shape (1, d) Containing the
            mean of the data set.
        cov: numpy.ndarray - shape (d, d) Containing the
            covariance matrix of the data set.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    mean = X.sum(axis=0)/X.shape[0]
    deviation = X - mean
    covariant = np.matmul(deviation.T, deviation)
    return mean[np.newaxis, ...], covariant/(X.shape[0]-1)
