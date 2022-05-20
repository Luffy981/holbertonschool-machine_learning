#!/usr/bin/env python3
"""Normalization constants"""
import numpy as np


def normalization_constants(X):
    """
        calculates the normalization (standartization)
        constants of a matrix.
        Args:
            X: numpy.ndarray - matrix to normalize
        returns:
            The mean and standard deviation
            of each feature respectively.
    """
    return np.mean(X, axis=0), np.std(X, axis=0)
