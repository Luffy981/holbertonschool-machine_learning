#!/usr/bin/env python3
"""
Module contains class for performing Bayesian Optimization on
a 1D gaussian process.
"""


import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process.
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Class constructor.

        Args:
            f: Black-box function to be optimized.
            X_init: numpy.ndarray - (t, 1) Inputs already sampled with the
            black-box function.
            Y_init: numpy.ndarray - (t, 1) Outputs of the black-box function
            for each input in X_init.
                t: Number of initial samples.
            bounds: tuple of (min, max) representing the bounds of the space
            in which to look for the optimal point.
            ac_samples: Number of samples that should be analyzed during
            acquisition.
            l: Length parameter for the kernel.
            sigma_f: Standard deviation given to the output of the black-box
            function.
            xsi: Exploration-exploitation factor for acquisition.
            minimize: bool determining whether optimization should be performed
            for minimization (True) or maximization (False).

        Public Instance Attributes:
            f: Black-box function.
            gp: Instance of the class GaussianProcess.
            X_s: numpy.ndarray - (ac_samples, 1) Acquisition sample points,
            evenly spaced between min and max.
            xsi: Exploration-exploitation factor.
            minimize: Bool for minimization versus maximization.
        """
        START, STOP = bounds

        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(START, STOP, num=ac_samples)[..., np.newaxis]
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location using Expected Improvement
        aquisition function.

        Return: X_next, EI
            X_next: numpy.ndarray - (1,) Next best sample point.
            EI: numpy.ndarray - (ac_samples,) Expected improvement of each
            potential sample.
        """

        mu, _ = self.gp.predict(self.gp.X)
        sample_mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            mu_opt = np.min(mu)
        else:
            mu_opt = np.max(mu)

        numZ = mu_opt - sample_mu - self.xsi
        Z = numZ / sigma
        EI = ((numZ * norm.cdf(Z)) + (sigma * norm.pdf(Z)))
        EI[sigma == 0.0] = 0.0

        return self.X_s[np.argmax(EI)], np.array(EI)
