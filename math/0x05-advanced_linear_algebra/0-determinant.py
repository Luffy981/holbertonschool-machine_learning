#!/usr/bin/env python3
"""
Determinant of a matrix
"""


def mydeepcopy(L):
    """
    Deep copy
    """
    if isinstance(L, list):
        ret = []
        for i in L:
            ret.append(mydeepcopy(i))
    elif isinstance(L, (int, float, type(None), str, bool)):
        ret = L
    else:
        raise ValueError("Unexpected type for mydeepcopy function")
    return ret


def determinant(matrix):
    """
    Args:
        matrix: is a list of lists whose determinant should be calculated
    Returns:
        the determinant of matrix
    """
    if matrix and matrix == [[]]:
        return 1
    for item in matrix:
        if type(item) is not list:
            raise TypeError("matrix must be a list of lists")
    for vec in matrix:
        if len(vec) != len(matrix):
            raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]
    elif len(matrix) == 2 and len(matrix[0]) == 2:
        return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
    else:
        result = 0
        for index, value in enumerate(matrix[0]):
            copia = mydeepcopy(matrix)
            for ele in copia[1:]:
                del ele[index]
                # print(copia)
            if index % 2 != 0:
                result += - value * determinant(copia[1:])
            else:
                result += value * determinant(copia[1:])
    return result
