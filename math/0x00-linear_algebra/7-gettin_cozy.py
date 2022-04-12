#!/usr/bin/env python3
import copy
def cat_matrices2D(mat1, mat2, axis=0):
    """
    concatenates two matrices along a specific axis
    """
    mat1_copy = copy.deepcopy(mat1)
    mat2_copy = copy.deepcopy(mat2)
    if axis == 0:
        for row in mat2_copy:
            mat1_copy = mat1_copy + [row]
        print("NO AXIS")
        return mat1_copy
    for i in range(len(mat2_copy)):
        mat1_copy[i] = mat1_copy[i] + mat2_copy[i]
    return mat1_copy
