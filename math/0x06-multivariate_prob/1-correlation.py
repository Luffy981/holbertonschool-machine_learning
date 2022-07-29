#!/usr/bin/env python3
"""
   Module contains function for
   computing correlation matrix.
"""


import numpy as np
from numpy.core.numeric import outer


def correlation(C):
    """
        Calculates a correlation matrix.

        Args:
        C: numpy.ndarray - Covariant matrix.

        Return:
        numpy.ndarray - Correlation matrix.
    """

    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")

    shp = C.shape
    if len(shp) < 2 or shp[0] != shp[1]:
        raise ValueError("C must be a 2D square matrix")

    Di = np.sqrt(np.diag(C))
    outer_Di = np.outer(Di, Di)
    corr = C / outer_Di
    corr[C == 0] = 0
    return corr
