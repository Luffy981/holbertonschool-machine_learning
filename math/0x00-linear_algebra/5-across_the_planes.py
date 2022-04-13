#!/usr/bin/env python3
"""
add matrix
"""


def add_matrices2D(mat1, mat2):
    """
    add matrix 2D
    """
    if len(mat1[0]) != len(mat2[0]):
        return None
    sum = []
    for row in range(len(mat1)):
        lit = []
        for i in range(len(mat1[0])):
            lit.append(mat1[row][i] + mat2[row][i])
        sum.append(lit)
    return sum
