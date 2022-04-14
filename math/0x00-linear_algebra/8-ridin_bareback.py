#!/usr/bin/env python3
"""multiplication matrix"""


def mat_mul(mat1, mat2):
    """
    performs matrix multiplication
    """
    mul = []
    if len(mat1[0]) != len(mat2):
        return None
    for i in range(len(mat1)):
        lit = []
        for k in range(len(mat2[0])):
            sum = 0
            for j in range(len(mat2)):
                sum += mat1[i][j] * mat2[j][k]
            lit.append(sum)
        mul.append(lit)
    return mul
