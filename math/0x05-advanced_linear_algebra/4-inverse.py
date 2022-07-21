#!/usr/bin/env python3
"""
Function that calculates the inverse matrix of a matrix
"""


def determinant(matrix):
    """
    matrix is a list of lists whose determinant should be calculated
    If matrix is not a list of lists, raise a TypeError with the
    message matrix must be a list of lists
    If matrix is not square, raise a ValueError with the message
    matrix must be a square matrix
    The list [[]] represents a 0x0 matrix
    Returns: the determinant of matrix
    """
    if type(matrix) is not list or len(matrix) is 0:
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        return 1
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(row) is 0 and len(matrix) is 1:
            return 1
        if len(row) != len(matrix):
            raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return (matrix[0][0] *
                matrix[1][1]) - (matrix[0][1] *
                                 matrix[1][0])
    cof = 1
    d = 0
    for i in range(len(matrix)):
        element = matrix[0][i]
        sub_matrix = []
        for row in range(len(matrix)):
            if row == 0:
                continue
            new_row = []
            for column in range(len(matrix)):
                if column == i:
                    continue
                new_row.append(matrix[row][column])
            sub_matrix.append(new_row)
        d += (element * cof * determinant(sub_matrix))
        cof *= -1
    return (d)


def cofactor(matrix):
    """
    matrix is a list of lists whose minor matrix should be calculated
    If matrix is not a list of lists, raise a TypeError with the message
    matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError with the
    message matrix must be a non-empty square matrix
    Returns: the minor matrix of matrix
    """
    if type(matrix) is not list or len(matrix) is 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(row) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) is 1:
        return [[1]]
    cof = 1
    cof_mat = []
    for i in range(len(matrix)):
        cof_row = []
        for column_i in range(len(matrix)):
            new_matrix = []
            for row in range(len(matrix)):
                if row == i:
                    continue
                new_row = []
                for column in range(len(matrix)):
                    if column == column_i:
                        continue
                    new_row.append(matrix[row][column])
                new_matrix.append(new_row)
            cof_row.append(cof * determinant(new_matrix))
            cof *= -1
        cof_mat.append(cof_row)
        if len(matrix) % 2 is 0:
            cof *= -1
    return cof_mat


def adjugate(matrix):
    """
    matrix is a list of lists whose adjugate matrix should be calculated
    If matrix is not a list of lists, raise a TypeError with the message
    matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError with the
    message matrix must be a non-empty square matrix
    Returns: the adjugate matrix of matrix
    """
    if type(matrix) is not list or len(matrix) is 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(row) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) is 1:
        return [[1]]
    adj = cofactor(matrix)
    transp = []
    for j in range(len(adj[0])):
        mat = []
        for i in range(len(adj)):
            mat.append(adj[i][j])
        transp.append(mat)
    return transp


def inverse(matrix):
    """
    matrix is a list of lists whose inverse should be calculated
    If matrix is not a list of lists, raise a TypeError with the
    message matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError with
    the message matrix must be a non-empty square matrix
    Returns: the inverse of matrix, or None if matrix is singular
    """
    if type(matrix) is not list or len(matrix) is 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(row) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")
    adj = adjugate(matrix)
    det = determinant(matrix)
    if det == 0:
        return None
    inv = []
    for i in range(len(matrix)):
        mat = []
        for j in range(len(matrix[0])):
            mat.append(adj[i][j] / det)
        inv.append(mat)
    return inv1
