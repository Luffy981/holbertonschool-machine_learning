#!/usr/bin/env python3
def np_elementwise(mat1, mat2):
    """
    performs element-wise addition, subtraction, multiplication, and division
    """
    sum = mat1 + mat2
    sub = mat1 - mat2
    mult = mat1 * mat2
    div = mat1 / mat2
    return(sum, sub, mult, div)
