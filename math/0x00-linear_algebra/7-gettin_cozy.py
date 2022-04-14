#!/usr/bin/env python3
"""concatenate matrix 2D"""


def mydeepcopy(L):
    """deepcopy"""
    if isinstance(L, list):
        ret = []
        for i in L:
            ret.append(mydeepcopy(i))
    elif isinstance(L, (int, float, type(None), str, bool)):
        ret = L
    else:
        raise ValueError("Unexpected type for mydeepcopy function")

    return ret


def cat_matrices2D(mat1, mat2, axis=0):
    """
    concatenates two matrices along a specific axis
    """
    mat1_copy = mydeepcopy(mat1)
    mat2_copy = mydeepcopy(mat2)
    if len(mat1[0]) != len(mat2[0])
        return None
    if axis == 0:
        for row in mat2_copy:
            mat1_copy = mat1_copy + [row]
        return mat1_copy
    for i in range(len(mat2_copy)):
        mat1_copy[i] = mat1_copy[i] + mat2_copy[i]
    return mat1_copy
