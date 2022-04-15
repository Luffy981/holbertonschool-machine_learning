#!/usr/bin/env python3
""" Concat matrices """


def is_list(matrix):
    """ returns True if the matrix is a list """
    return type(matrix) is list


def matrix_shape(matrix):
    """ Returns the shape of the matrix"""
    if type(matrix) != list or not matrix:
        return [0, ]
    return [len(matrix), *matrix_shape(matrix[0])]


def equal_without(matrix1, matrix2, without=0, index=0):
    """Returns true if the two matrices are equal without any index"""

    if without != -1:
        del matrix1[without]
        del matrix2[without]
        without = -1

    if index >= len(matrix1):
        return True

    try:
        if matrix1[index] != matrix2[index]:
            return False
    except IndexError:
        return True

    return equal_without(matrix1, matrix2, without, index + 1)


def cat_matrices2D(mat1, mat2, axis=0, firts=True):
    """ Concat matrices to a single matrix """
    if not is_list(mat1) and not is_list(mat2):
        return None

    shape_one = matrix_shape(mat1)
    shape_two = matrix_shape(mat2)
    if firts and not equal_without(shape_one, shape_two, axis):
        return None

    if (axis == 0):
        return [*mat1, *mat2]

    result = list(range(len(mat1)))
    for i in range(len(mat1)):
        result[i] = cat_matrices2D(mat1[i], mat2[i], axis - 1, False)
    return result


def test_one():
    """ test one and one of the matrices"""
    mat1 = [[1, 2], [3, 4]]
    mat2 = [[5, 6]]
    mat3 = [[7], [8]]
    mat5 = cat_matrices2D(mat1, mat3, axis=1)
    print(mat5)
    mat4 = cat_matrices2D(mat1, mat2)
    print(mat4)
    mat1[0] = [9, 10]
    mat1[1].append(5)
    print(mat1)
    print(mat4)
    print(mat5)


def test_two():
    """ test one and two matrices"""
    m1 = [[4, -7, 56, 2], [5, 106, 7, 2]]
    m2 = [[2, -6, 3], [0, -6, 3]]
    m = cat_matrices2D(m1, m2)
    print(m)
    m1 = [[484, 247], [554, 16], [5, 88]]
    m2 = [[233, -644, 325], [406, -16, 33], [765, 34, -39]]
    m = cat_matrices2D(m1, m2, axis=0)
    print(m)
    m1 = [[-54, -87, 95], [54, 16, -72]]
    m2 = [[12, 63, 79], [-10, 69, -9], [76, 45, -11]]
    m = cat_matrices2D(m1, m2, axis=1)
    print(m)


def test_three():
    """ test one and two matrices"""
    m1 = [[], []]
    m2 = [[], [], []]
    m = cat_matrices2D(m1, m2)
    exist = m is m1 or m is m2
    if not is_list(m) or exist or not len(m) or not is_list(m[0]):
        print("Not a new matrix")
    print(m)
    m1 = [[], [], []]
    m2 = [[], []]
    m = cat_matrices2D(m1, m2, axis=0)
    exist = m is m1 or m is m2
    if not is_list(m) or exist or not len(m) or not is_list(m[0]):
        print("Not a new matrix")
    print(m)
    m1 = [[], [], []]
    m2 = [[], [], []]
    m = cat_matrices2D(m1, m2, axis=1)
    exist = m is m1 or m is m2
    if not is_list(m) or exist or not len(m) or not is_list(m[0]):
        print("Not a new matrix")
    print(m)


def test_four():
    """" Test three matrices with different matrices"""
    m1 = [[51, 24, 73], [93, 45, 77]]
    m2 = [[], [], []]
    m = cat_matrices2D(m1, m2)
    print(m)
    m2 = [[51, 24, 73], [93, 45, 77]]
    m1 = [[], [], []]
    m = cat_matrices2D(m1, m2, axis=0)
    print(m)
    m1 = [[75, 23, 58], [32, 5, 67], [34, 65, 22]]
    m2 = [[], [], []]
    m = cat_matrices2D(m1, m2, axis=1)
    exist = m is m1 or m is m2
    if not is_list(m) or exist or not len(m) or not is_list(m[0]):
        print("Not a new matrix")
    print(m)


if __name__ == '__main__':
    test_one()
    test_two()
    test_three()
    test_four()
