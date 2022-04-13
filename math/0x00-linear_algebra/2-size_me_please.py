#!/usr/bin/env python3
def matrix_shape(matrix):
    """
    size matrix
    """
    try:
        if matrix[0][0][0]:
            return [len(matrix), len(matrix[0]), len(matrix[0][0])]
    except Exception:
        pass
    return [len(matrix), len(matrix[0])]
