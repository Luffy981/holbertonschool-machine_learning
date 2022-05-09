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
        one_hot = np.eye(classes)[Y]
    except Exception:
        return None
    return one_hot
