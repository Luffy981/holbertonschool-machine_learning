#!/usr/bin/env python3
"""function shape"""


def matrix_shape(matrix):
    """
    size matrix
    """
    shape = []
    while(type(matrix) is list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
