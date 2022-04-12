#!/usr/bin/env python3
def matrix_transpose(matrix):
    transpose = []
    for i in range(len(matrix[0])):
        litle = []
        for row in matrix:
            litle.append(row[i])
        transpose.append(litle)
    return transpose
