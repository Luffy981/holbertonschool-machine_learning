#!/usr/bin/env python3
"""
Module contains GaussianProcess class that represents
a noiseless 1D Gaussian Process.
"""


import numpy as np


class GaussianProcess():
    """
    Class that represents a noiseless 1D Gaussian Process.
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor.

        Args:
            X_init: numpy.ndarray - (t, 1) representing the inputs already
            sampled with the black-box function.
            Y_init: numpy.ndarray - (t, 1) representing the outputs of the
            black-box function for each input in X_init.
            t: Number of initial samples.
            length: Length parameter for the kernel.
            sigma_f: Standard deviation given to the output of the black-box
            function.

        Attributes:
            Sets the public instance attributes X, Y, l, and sigma_f
            corresponding to the respective constructor inputs.
            Sets the public instance attribute K, representing the current
            covariance kernel matrix for the Gaussian process.
        """

        self.X = X_init
        self.Y = Y_init
        self.t = X_init.shape[0]
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Public instance method that calculates the covariance kernel matrix
        between two matrices using the Radial Basis Function (RBF).

        Args:
            X1: numpy.ndarray - (m, 1)
            X2: numpy.ndarray - (n, 1)

        Return:
            Covariance kernel matrix as a numpy.ndarray - (m, n).
        """
        cov = np.exp(-((X1 - X2.T)**2) / (2 * (self.l**2)))
        return (self.sigma_f**2) * cov

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation of points in a
        Gaussian process.

        Args:
            X_s: numpy.ndarray - (s, 1) All of the points whose mean and
            standard deviation should be calculated.
                s: Number of sample points.

        Return: mu, sigma
            mu: numpy.ndarray - (s,) Mean for each point in X_s.
            sigma: numpy.ndarray - (s,) Variance for each point in X_s.
        """
        s = X_s.size

        X1X2 = self.kernel(self.X, X_s)
        solution = np.linalg.solve(self.K, X1X2).T
        X2X2 = self.kernel(X_s, X_s)
        mu = solution @ self.Y
        sigma = X2X2 - (solution @ X1X2)

        return mu.reshape(s,), np.diag(sigma)
