#!/usr/bin/env python3
"""
PCA on a dataset
"""
import numpy as np


def pca(X, ndim):
    """
    Args:
        X is a numpy.ndarray of shape (n, d) where:
           n is the number of data points
           d is the number of dimensions in each point
        ndim is the new dimensionality of the transformed X
    Returns:
        T, a numpy.ndarray of shape (n, ndim) containing
         the transformed version of X
    """
    X_meaned = X - np.mean(X, axis=0)
    U, S, V = np.linalg.svd(X_meaned)
    W = (V.T)[:, :ndim]
    return X_meaned.dot(W)
