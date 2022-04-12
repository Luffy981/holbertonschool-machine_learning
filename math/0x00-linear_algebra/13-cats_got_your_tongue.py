#!/usr/bin/env python3
import numpy as np
def np_cat(mat1, mat2, axis=0):
    """
    concatenate with numpy
    """
    return np.concatenate((mat1, mat2), axis=axis)
