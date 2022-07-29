#!/usr/bin/env python3
"""Module contains MultiNormal Class."""


import numpy as np


class MultiNormal():
    """
       Class which represents multivariate normal
        distribution.
    """

    def __init__(self, data):
        """
        Class Constructor

        Args:
            data: numpy.ndarray of shape (d, n)
                n: Number of data points.
                d: Number of dimensions in each data point.

        Public Attributes:
            mean: numpy.ndarray of shape (d, 1)
            cov: numpy.ndarray of shape (d, d
        """
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        self.mean, self.cov = self.mean_cov(data.T)

    @staticmethod
    def mean_cov(X):
        """
        Calculates the mean and covariance of
            a dataset.

        Args:
            X: numpy.ndarray - shape (n, d) containing data set.
                n: Number of data points.
                d: Number of dimensions in each data point.
        Returns: mean, cov:
            mean: numpy.ndarray - shape (d, 1) containing the
                mean of the data set.
            cov: numpy.ndarray - shape (d, d) containing the
                covariance matrix of the data set.
        """
        if type(X) is not np.ndarray or len(X.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        if X.shape[0] < 2:
            raise ValueError("data must contain multiple data points")

        n, d = X.shape
        mean = X.sum(axis=0)/n
        deviation = X - mean
        covariant = np.matmul(deviation.T, deviation)
        return mean.reshape((d, 1)), covariant/(X.shape[0]-1)

    @staticmethod
    def correlation(C):
        """
        Calculates a correlation matrix.

        Args:
            C: numpy.ndarray - Covariant matrix.

        Return:
            numpy.ndarray - Correlation matrix.
        """

        Di = np.sqrt(np.diag(C))
        outer_Di = np.outer(Di, Di)
        corr = C / outer_Di
        corr[C == 0] = 0
        return corr

    def pdf(self, x):
        """
           Calculates multivariate pdf at data point.

           Args:
            x: numpy.ndarray - shape (d, 1) Data point.

           Return:
            Value of the pdf.
        """
        D = self.mean.shape[0]

        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")

        x1, x2 = x.shape

        if len(x.shape) != 2 or x1 != D or x2 != 1:
            raise ValueError(
                "x must have the shape ({}, 1)".format(D)
                )

        Px = (2*np.pi)**(D/2)
        Px = 1 / (Px * (np.linalg.det(self.cov)**(1/2)))
        covI = np.linalg.inv(self.cov)
        x_mu = x - self.mean
        dot = np.dot(np.dot(x_mu.T, covI), x_mu)
        return float(Px*np.exp((-1/2)*dot))
