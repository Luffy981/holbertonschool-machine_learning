#!/usr/bin/env python3
"""concat matrix"""


def cat_arrays(arr1, arr2):
    """concat matrix"""
    arr1_copy = arr1[:]
    arr2_copy = arr2[:]
    new_array = arr1_copy + arr2_copy
    return new_array
