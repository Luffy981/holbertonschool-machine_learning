#!/usr/bin/env python3
"""Normalization constants"""
import numpy as np


def normalize(X, m, s):
    """
        calculates the normalization (standartization)
        of a matrix.
        Args:
            X: numpy.ndarray - matrix to normalize
            m: numpy.ndarray - contains the mean of all
            features of X
            s: numpy.ndarray - contains the std of all
            features of X
        returns:
            The normalized X matrix
    """
    # Zscore = (X - mean) / stddev
    return (X - m) / s
