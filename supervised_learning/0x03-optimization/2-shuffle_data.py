#!/usr/bin/env python3
"""shuffle data"""
import numpy as np


def shuffle_data(X, Y):
    """
        shuffles the data points in two matrices the same way
        Args:
            X: numpy.ndarray - shape(m, nx) - matriz to shuffle
            Y: numpy.ndarray - shape(m, ny) - matriz to shuffle
    """
    # np.random.permutation: permute a sequence,
    # or return a permuted range.
    shuff = np.random.permutation(len(X))
    # shuff 2 matriz the same way
    return X[shuff], Y[shuff]
