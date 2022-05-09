#!/usr/bin/env python3
"""One hot encoding"""


import numpy as np


def one_hot_encode(Y, classes):
    """Converts a numeric label vector into a one-hot matrix"""
    if type(Y) is not np.ndarray:
        return None
    if type(classes) is not int:
        return None
    try:
        # The eye tool returns a 2-D array with 1’s as the diagonal
        # and 0’s elsewhere
        one_hot = np.eye(classes)[Y]
        return one_hot
    except Exception:
        return None
