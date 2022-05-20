#!/usr/bin/env python3
"""
batch normalization
"""


import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
       Normalizes an unactivated output of a neural network using
       batch normalization.
       Args:
         Z: numpy.ndarray - (m, n) that should be normalized
         gamma: numpy.ndarray - of shape (1, n) containing the
           scales used for batch normalization
         beta: numpy.ndarray - of shape (1, n) containing the
           offsets used for batch normalization
         epsilon: small number used to avoid division by zero
       Returns:
         The normalized Z matrix.
    """
    mean, variance = np.mean(Z, axis=0), np.std(Z, axis=0)**2
    Z_norm = (Z - mean)/((variance+epsilon)**(1/2.0))
    return gamma*Z_norm+beta
