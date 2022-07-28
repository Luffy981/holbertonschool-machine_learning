#!/usr/bin/env python3
"""
PCA on a dataset
"""
import numpy as np


def pca(X, var=0.95):
    """
    Args:
        X: is a numpy.ndarray of shape (n, d) where
            n is the number of data points
            d is the number of dimensions in each point
            all dimensions have a mean of 0 across all data points
        var is the fraction of the variance that the PCA transformation
        should maintain
    Returns:
         the weights matrix, W, that maintains var fraction of X‘s
         original variance
    """
    # SVD(Singular Value Deviation)
    # U = Unitary array(s).
    # vals = Vector(s) with the singular values,sorted in descending order.
    # eig = Unitary array(s). eigen vectors
    U, vals, eig = np.linalg.svd(X)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    eig = (eig.T)[:, idx]

    var_explained = []
    eig_sum = vals.sum()

    for i in range(vals.shape[0]):
        var_explained.append(vals[i]/eig_sum)

    # Cumulatice sum
    Csum = np.cumsum(var_explained)

    for i in range(Csum.shape[0]):
        if Csum[i] >= var:
            return eig[:, :i+1]
    return eig
