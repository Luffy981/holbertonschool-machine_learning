#!/usr/bin/env python3
"""Matrix Definitness Module"""


import numpy as np


def definiteness(matrix):
    """
       Computes definitness of a matrix.

       Args:
         matrix: Matrix to compute.

       Return:
         Definitness of a matrix.
    """

    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")

    try:
        eig, _ = np.linalg.eig(matrix)

        pos = (eig > 0)
        neg = (eig < 0)

        if False not in pos:
            return "Positive definite"
        elif False not in neg:
            return "Negative definite"
        elif True in pos and True not in neg:
            return "Positive semi-definite"
        elif True in neg and True not in pos:
            return "Negative semi-definite"
        return "Indefinite"
    except Exception:
        return None
